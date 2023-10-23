import os
import sys
import logging
from itertools import islice

import torch
from torch import distributed as dist
from torch import nn
from torch.utils import data as torch_data

from torchdrug import data, core, utils
from torchdrug.core import Registry as R
from torchdrug.utils import comm, pretty


module = sys.modules[__name__]
logger = logging.getLogger(__name__)


@R.register("core.MultiGraphEngine")
class MultiGraphEngine(core.Engine):

    def sample_edges_from_graph(self, batch):
        # sample graph id from 0 to k graphs in the model weighted by num edges
        probs = torch.tensor([getattr(self.model, f"fact_graph_{i}").num_edge for i in range(self.model.num_graphs)]).float()
        probs /= probs.sum()
        graph_id = torch.multinomial(probs, 1, replacement=False).item()
        #graph_id = torch.randint(0, self.model.num_graphs, (1,)).item()
        graph = getattr(self.model, f"fact_graph_{graph_id}")
        # sample training edges from this graph of bs size
        batch_size = len(batch)
        edge_mask = torch.randperm(graph.num_edge)[:batch_size]
        train_edges = graph.edge_list[edge_mask]
        return (train_edges, graph_id)

    def train(self, num_epoch=1, batch_per_epoch=None):
        """
        Train the model on different graphs, the custom collate function takes care of being sure that 
        each batch contains samples from only one graph.

        If ``batch_per_epoch`` is specified, randomly draw a subset of the training set for each epoch.
        Otherwise, the whole training set is used for each epoch.

        Parameters:
            num_epoch (int, optional): number of epochs
            batch_per_epoch (int, optional): number of batches per epoch
        """
        sampler = torch_data.DistributedSampler(self.train_set, self.world_size, self.rank)
        dataloader = data.DataLoader(self.train_set, self.batch_size, sampler=sampler, 
                                     num_workers=self.num_worker, collate_fn=self.sample_edges_from_graph,
                                    )
        batch_per_epoch = batch_per_epoch or len(dataloader)
        model = self.model
        model.split = "train"
        if self.world_size > 1:
            if self.device.type == "cuda":
                model = nn.parallel.DistributedDataParallel(model, device_ids=[self.device],
                                                            find_unused_parameters=True)
            else:
                model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.train()

        for epoch in self.meter(num_epoch):
            sampler.set_epoch(epoch)

            metrics = []
            start_id = 0
            # the last gradient update may contain less than gradient_interval batches
            gradient_interval = min(batch_per_epoch - start_id, self.gradient_interval)

            for batch_id, batch in enumerate(islice(dataloader, batch_per_epoch)):
                batch, graph_id = batch
                if self.device.type == "cuda":
                    batch = utils.cuda(batch, device=self.device)

                loss, metric = model((batch, graph_id))
                if not loss.requires_grad:
                    raise RuntimeError("Loss doesn't require grad. Did you define any loss in the task?")
                loss = loss / gradient_interval
                loss.backward()
                metrics.append(metric)

                if batch_id - start_id + 1 == gradient_interval:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    metric = utils.stack(metrics, dim=0)
                    metric = utils.mean(metric, dim=0)
                    if self.world_size > 1:
                        metric = comm.reduce(metric, op="mean")
                    self.meter.update(metric)

                    metrics = []
                    start_id = batch_id + 1
                    gradient_interval = min(batch_per_epoch - start_id, self.gradient_interval)

            if self.scheduler:
                self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, split, log=True):
        """
        Evaluate the model on all validation/test sets of all underlying graphs.

        Parameters:
            split (str): split to evaluate. Can be ``train``, ``valid`` or ``test``.
            log (bool, optional): log metrics or not

        Returns:
            dict: metrics
        """
        if comm.get_rank() == 0:
            logger.warning(pretty.separator)
            logger.warning("Evaluate on %s" % split)
        test_set = getattr(self, "%s_set" % split)
        all_metrics = []
        for graph_name, value in test_set.items():
            if comm.get_rank() == 0:
                logger.warning("Evaluate on %s" % graph_name)
            graph_id, edge_mask = value
            if self.device.type == "cuda":
                edge_mask = utils.cuda(edge_mask, device=self.device)
            graph = getattr(self.model, f"graph_{graph_id}")

            if len(edge_mask) == 2:
                test_edges = graph.edge_list[edge_mask[0]:edge_mask[1]]
            else:
                # for the fast_test case when IDs are a list, not an interval
                test_edges = graph.edge_list[edge_mask]
            sampler = torch_data.DistributedSampler(test_edges, self.world_size, self.rank)
            dataloader = data.DataLoader(test_edges, self.batch_size, sampler=sampler, num_workers=self.num_worker)
            model = self.model
            model.split = split

            model.eval()
            preds = []
            targets = []
            for batch in dataloader:
                if self.device.type == "cuda":
                    batch = utils.cuda(batch, device=self.device)

                pred, target = model.predict_and_target((batch, graph_id))
                preds.append(pred)
                targets.append(target)

            pred = utils.cat(preds)
            target = utils.cat(targets)
            if self.world_size > 1:
                pred = comm.cat(pred)
                target = comm.cat(target)
            metric = model.evaluate(pred, target)
            if log:
                self.meter.log(metric, category="%s/%s/epoch" % (graph_name, split))
            all_metrics.append(metric)
        
        avg_metric = {}
        for metric_name in all_metrics[0]:
            avg_metric[metric_name] = float(sum([m[metric_name].item() for m in all_metrics])) / len(all_metrics)
        return avg_metric
    
    # remove train/val/test indices from the config object to be sent to wandb
    def config_dict(self):

        self._config.pop('train_set')
        self._config.pop('valid_set')
        self._config.pop('test_set')
        return super().config_dict()
        
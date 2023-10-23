import math
import copy

import torch
from torch.nn import functional as F
from torch.utils import data as torch_data

#from ogb import linkproppred

from torchdrug import core, tasks, metrics, data
from torchdrug.layers import functional
from torchdrug.core import Registry as R

from torch_scatter import scatter_mean, scatter_add

#Evaluator = core.make_configurable(linkproppred.Evaluator)
#Evaluator = R.register("ogb.linkproppred.Evaluator")(Evaluator)
#setattr(linkproppred, "Evaluator", Evaluator)

# =============== KnowledgeGraphCompletion ===============

@R.register("tasks.KnowledgeGraphCompletionBase")
class KnowledgeGraphCompletionBase(tasks.KnowledgeGraphCompletion, core.Configurable):

    def __init__(self, *args, merge_penalize=False, penality_N2=10, **kwargs):
        super(KnowledgeGraphCompletionBase, self).__init__(*args, **kwargs)
        assert self.strict_negative
        self.merge_penalize = merge_penalize
        self.penality_N2 = penality_N2
    
    def preprocess(self, train_set, valid_set, test_set):
        if isinstance(train_set, torch_data.Subset):
            dataset = train_set.dataset
        else:
            dataset = train_set
        self.num_entity = dataset.num_entity
        self.num_relation = dataset.num_relation
        self.register_buffer("graph", dataset.graph)
        fact_mask = torch.ones(len(dataset), dtype=torch.bool)
        fact_mask[valid_set.indices] = 0
        fact_mask[test_set.indices] = 0
        if self.fact_ratio:
            length = int(len(train_set) * self.fact_ratio)
            index = torch.randperm(len(train_set))[length:]
            train_indices = torch.tensor(train_set.indices)
            fact_mask[train_indices[index]] = 0
            train_set = torch_data.Subset(train_set, index)
        self.register_buffer("fact_graph", dataset.graph.edge_mask(fact_mask))

        if self.sample_weight:
            degree_hr = torch.zeros(self.num_entity, self.num_relation, dtype=torch.long)
            degree_tr = torch.zeros(self.num_entity, self.num_relation, dtype=torch.long)
            for h, t, r in train_set:
                degree_hr[h, r] += 1
                degree_tr[t, r] += 1
            self.register_buffer("degree_hr", degree_hr)
            self.register_buffer("degree_tr", degree_tr)
        
        if hasattr(dataset, "merge_info"):
            self.merge_info = dataset.merge_info
            self.model.merge_info = dataset.merge_info

        return train_set, valid_set, test_set

    def _calculate_t_mask(self, graph, pos_h_index, pos_r_index):

        any = -torch.ones_like(pos_h_index)

        pattern = torch.stack([pos_h_index, any, pos_r_index], dim=-1)
        edge_index, num_t_truth = graph.match(pattern)
        t_truth_index = graph.edge_list[edge_index, 1]
        pos_index = torch.repeat_interleave(num_t_truth)
        #assert self.num_entity == graph.num_node # not true with inductive evaluation
        t_mask = torch.ones(len(pattern), graph.num_node, dtype=torch.bool, device=self.device)
        t_mask[pos_index, t_truth_index] = 0
        if hasattr(self, "merge_info"):
            assert len(self.merge_info) == 2, "only support merge for two datasets"
            gap = self.merge_info[0]["num_node"]
            indices = (pos_h_index < gap)
            t_mask[indices, gap:] = 0
            t_mask[~indices, :gap] = 0
        return t_mask

    def _calculate_h_mask(self, graph, pos_t_index, pos_r_index):
        any = -torch.ones_like(pos_t_index)

        pattern = torch.stack([any, pos_t_index, pos_r_index], dim=-1)
        edge_index, num_h_truth = graph.match(pattern)
        h_truth_index = graph.edge_list[edge_index, 0]
        pos_index = torch.repeat_interleave(num_h_truth)
        #assert self.num_entity == graph.num_node # not true with inductive evaluation
        h_mask = torch.ones(len(pattern), graph.num_node, dtype=torch.bool, device=self.device)
        h_mask[pos_index, h_truth_index] = 0
        if hasattr(self, "merge_info"):
            assert len(self.merge_info) == 2, "only support merge for two datasets"
            gap = self.merge_info[0]["num_node"]
            indices = (pos_t_index < gap)
            h_mask[indices, gap:] = 0
            h_mask[~indices, :gap] = 0
        return h_mask
    
    @torch.no_grad()
    def _strict_negative(self, pos_h_index, pos_t_index, pos_r_index):
        batch_size = len(pos_h_index)

        t_mask = self._calculate_t_mask(self.fact_graph, pos_h_index[:batch_size // 2], pos_r_index[:batch_size // 2])
        neg_t_candidate = t_mask.nonzero()[:, 1]
        num_t_candidate = t_mask.sum(dim=-1)
        neg_t_index = functional.variadic_sample(neg_t_candidate, num_t_candidate, self.num_negative)

        h_mask = self._calculate_h_mask(self.fact_graph, pos_t_index[batch_size // 2:], pos_r_index[batch_size // 2:])
        neg_h_candidate = h_mask.nonzero()[:, 1]
        num_h_candidate = h_mask.sum(dim=-1)
        neg_h_index = functional.variadic_sample(neg_h_candidate, num_h_candidate, self.num_negative)

        neg_index = torch.cat([neg_t_index, neg_h_index])

        return neg_index

    def target(self, batch):
        # test target
        batch_size = len(batch)
        pos_h_index, pos_t_index, pos_r_index = batch.t()

        t_mask = self._calculate_t_mask(self.graph, pos_h_index, pos_r_index)
        h_mask = self._calculate_h_mask(self.graph, pos_t_index, pos_r_index)

        mask = torch.stack([t_mask, h_mask], dim=1)
        target = torch.stack([pos_t_index, pos_h_index], dim=1)

        # in case of GPU OOM
        return mask.cpu(), target.cpu()

    def evaluate(self, pred, target):
        mask, target = target

        pos_pred = pred.gather(-1, target.unsqueeze(-1))
        if self.filtered_ranking:
            ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
        else:
            ranking = torch.sum(pos_pred <= pred, dim=-1) + 1
        print("mask_shape: ", mask.sum(-1).max())

        metric = {}
        for _metric in self.metric:
            if _metric == "mr":
                score = ranking.float().mean()
            elif _metric == "mrr":
                score = (1 / ranking.float()).mean()
            elif _metric.startswith("hits@"):
                threshold = int(_metric[5:])
                score = (ranking <= threshold).float().mean()
            else:
                raise ValueError("Unknown metric `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            metric[name] = score

        return metric
    def forward(self, batch, all_loss=None, metric=None):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)
        pos_h_index, pos_t_index, pos_r_index = batch.t()

        for criterion, weight in self.criterion.items():
            if criterion == "bce":
                target = torch.zeros_like(pred)
                target[:, 0] = 1
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

                neg_weight = torch.ones_like(pred)
                if self.adversarial_temperature > 0:
                    with torch.no_grad():
                        neg_weight[:, 1:] = F.softmax(pred[:, 1:] / self.adversarial_temperature, dim=-1)
                else:
                    neg_weight[:, 1:] = 1 / self.num_negative
                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)

            if self.sample_weight:
                sample_weight = self.degree_hr[pos_h_index, pos_r_index] * self.degree_tr[pos_t_index, pos_r_index]
                sample_weight = 1 / sample_weight.float().sqrt()
                loss = (loss * sample_weight).sum() / sample_weight.sum()
            
            loss = loss.mean()

            name = tasks._get_criterion_name(criterion)
            metric[name] = loss
            all_loss += loss * weight

        return all_loss, metric

@R.register("tasks.KnowledgeGraphCompletionAdapted")
class KnowledgeGraphCompletionAdapted(KnowledgeGraphCompletionBase, core.Configurable):

    def __init__(self, model, rel_models, criterion="bce", metric=("mr", "mrr", "hits@1", "hits@3", "hits@10", "mrr-tail", "hits@1-tail", "hits@10-tail"),
                 num_negative=128, margin=6, adversarial_temperature=0, strict_negative=True, filtered_ranking=True,
                 fact_ratio=None, sample_weight=True,
                 metric_per_rel=False, full_batch_eval=False):
        super(KnowledgeGraphCompletionAdapted, self).__init__(model, 
                 criterion=criterion, metric=metric,
                 num_negative=num_negative, margin=margin, 
                 adversarial_temperature=adversarial_temperature, 
                 strict_negative=strict_negative, filtered_ranking=filtered_ranking,
                 fact_ratio=fact_ratio, sample_weight=sample_weight)
        
        self.rel_models = rel_models
        self.metric_per_rel = metric_per_rel
        self.full_batch_eval = full_batch_eval
    
    def preprocess(self, train_set, valid_set, test_set):
        #ipdb.set_trace()
        train_set, valid_set, test_set = super(KnowledgeGraphCompletionAdapted, self).preprocess(train_set, valid_set, test_set)

        # process fact_graph
        rel_graphs = []
        for rel_model in self.rel_models:
            rel_graph = (rel_model).construct_relation_graph(self.fact_graph)
            rel_graphs.append(rel_graph)
        self.register_buffer("rel_graphs", data.Graph.pack(rel_graphs))

        return train_set, valid_set, test_set

    def predict(self, batch, all_loss=None, metric=None):
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        batch_size = len(batch)

        # get repr. for relations
        rel_inputs = []
        for i in range(len(self.rel_graphs)):
            # rel_input = self.rel_models[i](self.rel_graphs[i], None, all_loss=all_loss, metric=metric)["node_feature"]
            if self.rel_models[0].__class__.__name__ != "RelNBFNet":
                rel_input = self.rel_models[i](self.rel_graphs[i], None, all_loss=all_loss, metric=metric)["node_feature"]
            else:
                rel_input = self.rel_models[i](self.rel_graphs[i], None, pos_r_index, all_loss=all_loss, metric=metric)["node_feature"]
            
            rel_inputs.append(rel_input)

        if all_loss is None:
            # test
            all_index = torch.arange(self.num_entity, device=self.device)
            t_preds = []
            h_preds = []
            num_negative = self.num_entity if self.full_batch_eval else self.num_negative
            for neg_index in all_index.split(num_negative):
                r_index = pos_r_index.unsqueeze(-1).expand(-1, len(neg_index))
                h_index, t_index = torch.meshgrid(pos_h_index, neg_index)
                t_pred = self.model(self.fact_graph, rel_inputs, h_index, t_index, r_index, all_loss=all_loss, metric=metric)
                t_preds.append(t_pred)
            t_pred = torch.cat(t_preds, dim=-1)
            for neg_index in all_index.split(num_negative):
                r_index = pos_r_index.unsqueeze(-1).expand(-1, len(neg_index))
                t_index, h_index = torch.meshgrid(pos_t_index, neg_index)
                h_pred = self.model(self.fact_graph, rel_inputs, h_index, t_index, r_index, all_loss=all_loss, metric=metric)
                h_preds.append(h_pred)
            h_pred = torch.cat(h_preds, dim=-1)
            pred = torch.stack([t_pred, h_pred], dim=1)
            # in case of GPU OOM
            pred = pred.cpu()
        else:
            # train
            if self.strict_negative:
                neg_index = self._strict_negative(pos_h_index, pos_t_index, pos_r_index)
            else:
                neg_index = torch.randint(self.num_entity, (batch_size, self.num_negative), device=self.device)
            h_index = pos_h_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            t_index = pos_t_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            r_index = pos_r_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            t_index[:batch_size // 2, 1:] = neg_index[:batch_size // 2]
            h_index[batch_size // 2:, 1:] = neg_index[batch_size // 2:]
            pred = self.model(self.fact_graph, rel_inputs, h_index, t_index, r_index, all_loss=all_loss, metric=metric)

        return pred
    
    def target(self, batch):
        # test target
        batch_size = len(batch)
        pos_h_index, pos_t_index, pos_r_index = batch.t()

        t_mask = self._calculate_t_mask(self.graph, pos_h_index, pos_r_index)
        h_mask = self._calculate_h_mask(self.graph, pos_t_index, pos_r_index)

        mask = torch.stack([t_mask, h_mask], dim=1)
        target = torch.stack([pos_t_index, pos_h_index], dim=1)

        if self.metric_per_rel:
            rel = torch.stack([pos_r_index, pos_r_index+self.model.num_relation], dim=1)
            return mask.cpu(), target.cpu(), rel.cpu()

        # in case of GPU OOM
        return mask.cpu(), target.cpu()
    
    
    def predict_and_target(self, batch, all_loss=None, metric=None):
        pred, target = super(KnowledgeGraphCompletionAdapted, self).predict_and_target(batch, all_loss, metric)
        #return pred, target
        if self.graph.num_node > 5e4:
            ranking = self.get_ranking(pred, target)
            return ranking, ranking
        else:
            return pred, target

    def get_ranking(self, pred, target):
        mask, target = target
        pos_pred = pred.gather(-1, target.unsqueeze(-1))
        if self.filtered_ranking:
            ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
        else:
            ranking = torch.sum(pos_pred <= pred, dim=-1) + 1

        return ranking

    def evaluate(self, pred, target):
        
        if not isinstance(target, torch.Tensor):
            ranking = self.get_ranking(pred, target)
        else:
            ranking = pred

        metric = {}
        for _metric in self.metric:
            # e.g. mrr-tail means mrr for tail prediction
            if "-" in _metric:
                _metric_name, direction = _metric.split("-")
                if direction == "head":
                    _ranking = ranking.select(1, 1)
                elif direction == "tail":
                    _ranking = ranking.select(1, 0)
                else:
                    raise ValueError("Unknown direction `%s`" % direction)
            else:
                _ranking = ranking
                _metric_name = _metric
            if _metric_name == "mr":
                score = _ranking.float().mean()
            elif _metric_name == "mrr":
                score = (1 / _ranking.float()).mean()
            elif _metric_name.startswith("hits@"):
                threshold = int(_metric_name[5:])
                score = (_ranking <= threshold).float().mean()
            else:
                raise ValueError("Unknown metric `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            metric[name] = score

        return metric


# =============== InductiveKnowledgeGraphCompletion ===============

@R.register("tasks.InductiveKnowledgeGraphCompletion")
class InductiveKnowledgeGraphCompletion(KnowledgeGraphCompletionBase, core.Configurable):

    def __init__(self, model, toy_eval=False, criterion="bce", metric=("mr", "mrr", "hits@1", "hits@3", "hits@10", "hits@10_50"),
                 num_negative=128, margin=6, adversarial_temperature=0, strict_negative=True, sample_weight=True,
                 metric_per_rel=False):
        super(InductiveKnowledgeGraphCompletion, self).__init__(
            model, criterion, metric, num_negative, margin, adversarial_temperature, strict_negative,
            sample_weight=sample_weight)
        
        self.toy_eval = toy_eval
        self.metric_per_rel = metric_per_rel

    def preprocess(self, train_set, valid_set, test_set):
        if isinstance(train_set, torch_data.Subset):
            dataset = train_set.dataset
        else:
            dataset = train_set
        self.num_entity = dataset.num_entity
        self.num_relation = dataset.num_relation

        if self.sample_weight:
            degree_hr = torch.zeros(self.num_entity, self.num_relation, dtype=torch.long)
            degree_tr = torch.zeros(self.num_entity, self.num_relation, dtype=torch.long)
            for h, t, r in train_set:
                degree_hr[h, r] += 1
                degree_tr[t, r] += 1
            self.register_buffer("degree_hr", degree_hr)
            self.register_buffer("degree_tr", degree_tr)

        self.register_buffer("fact_graph", dataset.train_graph)
        self.register_buffer("train_graph", dataset.train_graph)
        self.register_buffer("valid_graph", dataset.valid_graph)
        self.register_buffer("test_graph", dataset.test_graph)
        self.register_buffer("graph", dataset.graph)
        self.register_buffer("inductive_graph", dataset.inductive_graph)

        return train_set, valid_set, test_set

    def predict(self, batch, all_loss=None, metric=None):
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        batch_size = len(batch)
        fact_graph = getattr(self, "%s_graph" % self.split)

        if all_loss is None:
            # test
            all_index = torch.arange(fact_graph.num_node, device=self.device)
            t_preds = []
            h_preds = []
            for neg_index in all_index.split(self.num_negative):
                r_index = pos_r_index.unsqueeze(-1).expand(-1, len(neg_index))
                h_index, t_index = torch.meshgrid(pos_h_index, neg_index)
                t_pred = self.model(fact_graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric)
                t_preds.append(t_pred)
            t_pred = torch.cat(t_preds, dim=-1)
            for neg_index in all_index.split(self.num_negative):
                r_index = pos_r_index.unsqueeze(-1).expand(-1, len(neg_index))
                t_index, h_index = torch.meshgrid(pos_t_index, neg_index)
                h_pred = self.model(fact_graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric)
                h_preds.append(h_pred)
            h_pred = torch.cat(h_preds, dim=-1)
            pred = torch.stack([t_pred, h_pred], dim=1)
            # in case of GPU OOM
            pred = pred.cpu()
        else:
            # train
            if self.strict_negative:
                neg_index = self._strict_negative(pos_h_index, pos_t_index, pos_r_index)
            else:
                neg_index = torch.randint(self.num_entity, (batch_size, self.num_negative), device=self.device)
            h_index = pos_h_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            t_index = pos_t_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            r_index = pos_r_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            t_index[:batch_size // 2, 1:] = neg_index[:batch_size // 2]
            h_index[batch_size // 2:, 1:] = neg_index[batch_size // 2:]
            pred = self.model(fact_graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric)

        return pred

    def target(self, batch):
        # test target
        batch_size = len(batch)
        if self.split == "train":
            graph = self.graph
        elif self.split == "valid": # "valid" and use_inductive_valid = no
            flag = False
            if (self.train_graph.edge_list.shape[0] == self.valid_graph.edge_list.shape[0]):
                if (self.train_graph.edge_list == self.valid_graph.edge_list).all():
                    flag = True
            graph = self.graph if flag else self.inductive_graph
        else:
            graph = self.inductive_graph
        pos_h_index, pos_t_index, pos_r_index = batch.t()

        t_mask = self._calculate_t_mask(graph, pos_h_index, pos_r_index)
        h_mask = self._calculate_h_mask(graph, pos_t_index, pos_r_index)

        mask = torch.stack([t_mask, h_mask], dim=1)
        target = torch.stack([pos_t_index, pos_h_index], dim=1)

        if self.metric_per_rel:
            rel = torch.stack([pos_r_index, pos_r_index], dim=1)
            return mask.cpu(), target.cpu(), rel.cpu()

        # in case of GPU OOM
        return mask.cpu(), target.cpu()

    def evaluate(self, pred, target):
        #ipdb.set_trace()
        if self.metric_per_rel:
            mask, target, rel = target
            rel = rel.reshape(-1,)
        else:
            mask, target = target

        pos_pred = pred.gather(-1, target.unsqueeze(-1))
        ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
        #ipdb.set_trace()
        if self.toy_eval:
            weights = torch.ones(pred.shape[-1]).expand(pred.shape[0] * pred.shape[1], -1)
            weights = weights * mask.reshape(-1, pred.shape[-1])
            idx = torch.multinomial(weights, num_samples=50, replacement=False)
            idx = idx.reshape(pred.shape[0], pred.shape[1], -1)
            #ranking = torch.sum(((pos_pred <= pred) & mask).gather(2, idx), dim=-1) + 1
            neg_pred = pred.gather(2, idx)
            
            optimistic_ranks = (pos_pred < neg_pred).sum(dim=-1)
            pessimistic_ranks = (pos_pred <= neg_pred).sum(dim=-1)
            ranking = 0.5 * (optimistic_ranks + pessimistic_ranks) + 1

        metric = {}
        for _metric in self.metric:
            if _metric == "mr":
                score = ranking.float()
            elif _metric == "mrr":
                score = (1 / ranking.float())
            elif _metric.startswith("hits@"):
                values = _metric[5:].split("_")
                threshold = int(values[0])
                if len(values) > 1:
                    num_sample = int(values[1])
                    # unbiased estimation: old version
                    fp_rate = (ranking - 1).float() / mask.sum(dim=-1)
                    if self.toy_eval:
                        assert num_sample == 50
                        fp_rate = (ranking - 1).float() / 51
                    score = 0
                    for i in range(threshold):
                        # choose i false positive from num_sample negatives
                        num_comb = math.factorial(num_sample) / math.factorial(i) / math.factorial(num_sample - i)
                        score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i))
                    #score = score
                else:
                    score = (ranking <= threshold).float()
            else:
                raise ValueError("Unknown metric `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            if self.metric_per_rel:
                score = score.reshape(-1,)
                score_per_rel = scatter_mean(score, rel, dim_size=self.model.num_relation * 2) # undirected!
                for ridx in range(self.model.num_relation * 2):
                    metric[name+"_rel_%d" % ridx] = score_per_rel[ridx]

            score = score.mean()
            metric[name] = score

        return metric

@R.register("tasks.InductiveKnowledgeGraphCompletionAdapted")
class InductiveKnowledgeGraphCompletionAdapted(InductiveKnowledgeGraphCompletion, core.Configurable):

    def __init__(self, model, rel_models, toy_eval=False, criterion="bce", metric=("mr", "mrr", "hits@1", "hits@3", "hits@10", "hits@10_50"),
                 num_negative=128, margin=6, adversarial_temperature=0, strict_negative=True, sample_weight=True,
                 metric_per_rel=False, full_batch_eval=False):
        super(InductiveKnowledgeGraphCompletionAdapted, self).__init__(
            model, toy_eval=toy_eval, criterion=criterion, metric=metric, 
            num_negative=num_negative, margin=margin, adversarial_temperature=adversarial_temperature, 
            strict_negative=strict_negative, sample_weight=sample_weight, metric_per_rel=metric_per_rel)
        
        self.rel_models = rel_models
        self.full_batch_eval = full_batch_eval

    def preprocess(self, train_set, valid_set, test_set):
        if isinstance(train_set, torch_data.Subset):
            dataset = train_set.dataset
        else:
            dataset = train_set
        self.num_entity = dataset.num_entity
        self.num_relation = dataset.num_relation

        if self.sample_weight:
            degree_hr = torch.zeros(self.num_entity, self.num_relation, dtype=torch.long)
            degree_tr = torch.zeros(self.num_entity, self.num_relation, dtype=torch.long)
            for h, t, r in train_set:
                degree_hr[h, r] += 1
                degree_tr[t, r] += 1
            self.register_buffer("degree_hr", degree_hr)
            self.register_buffer("degree_tr", degree_tr)

        self.register_buffer("fact_graph", dataset.train_graph)
        self.register_buffer("train_graph", copy.copy(dataset.train_graph))
        self.register_buffer("valid_graph", dataset.valid_graph)
        self.register_buffer("test_graph", dataset.test_graph)
        self.register_buffer("graph", dataset.graph)
        self.register_buffer("inductive_graph", dataset.inductive_graph)

        train_rel_graphs, valid_rel_graphs, test_rel_graphs = [], [], []
        for rel_model in self.rel_models:

            # process fact_graph
            # we do undirected transformation inside the rel_graph construction function
            rel_graph = rel_model.construct_relation_graph(self.train_graph) 
            train_rel_graphs.append(rel_graph)
            rel_graph = rel_model.construct_relation_graph(self.valid_graph)
            valid_rel_graphs.append(rel_graph)
            rel_graph = rel_model.construct_relation_graph(self.test_graph)
            test_rel_graphs.append(rel_graph)
        
        self.register_buffer("train_rel_graphs", data.Graph.pack(train_rel_graphs))
        self.register_buffer("valid_rel_graphs", data.Graph.pack(valid_rel_graphs))
        self.register_buffer("test_rel_graphs", data.Graph.pack(test_rel_graphs))

        #ipdb.set_trace()

        return train_set, valid_set, test_set

    def predict(self, batch, all_loss=None, metric=None):
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        batch_size = len(batch)
        fact_graph = getattr(self, "%s_graph" % self.split)

        rel_graphs = getattr(self, "%s_rel_graphs" % self.split)

        rel_inputs = []
        for i in range(len(rel_graphs)):
            #rel_input = self.rel_models[i](rel_graphs[i], None)["node_feature"]
            if self.rel_models[0].__class__.__name__ != "RelNBFNet":
                rel_input = self.rel_models[i](rel_graphs[i], None, all_loss=all_loss, metric=metric)["node_feature"]
            else:
                rel_input = self.rel_models[i](rel_graphs[i], None, pos_r_index, all_loss=all_loss, metric=metric)["node_feature"]
            
            rel_inputs.append(rel_input)

        if all_loss is None:
            # test
            all_index = torch.arange(fact_graph.num_node, device=self.device)
            t_preds = []
            h_preds = []
            num_negative = fact_graph.num_node if self.full_batch_eval else self.num_negative
            for neg_index in all_index.split(num_negative):
                r_index = pos_r_index.unsqueeze(-1).expand(-1, len(neg_index))
                h_index, t_index = torch.meshgrid(pos_h_index, neg_index)
                t_pred = self.model(fact_graph, rel_inputs, h_index, t_index, r_index, all_loss=all_loss, metric=metric)
                t_preds.append(t_pred)
            t_pred = torch.cat(t_preds, dim=-1)
            for neg_index in all_index.split(num_negative):
                r_index = pos_r_index.unsqueeze(-1).expand(-1, len(neg_index))
                t_index, h_index = torch.meshgrid(pos_t_index, neg_index)
                h_pred = self.model(fact_graph, rel_inputs, h_index, t_index, r_index, all_loss=all_loss, metric=metric)
                h_preds.append(h_pred)
            h_pred = torch.cat(h_preds, dim=-1)
            pred = torch.stack([t_pred, h_pred], dim=1)
            # in case of GPU OOM
            pred = pred.cpu()
        else:
            # train
            if self.strict_negative:
                neg_index = self._strict_negative(pos_h_index, pos_t_index, pos_r_index)
            else:
                neg_index = torch.randint(self.num_entity, (batch_size, self.num_negative), device=self.device)
            h_index = pos_h_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            t_index = pos_t_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            r_index = pos_r_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            t_index[:batch_size // 2, 1:] = neg_index[:batch_size // 2]
            h_index[batch_size // 2:, 1:] = neg_index[batch_size // 2:]
            pred = self.model(fact_graph, rel_inputs, h_index, t_index, r_index, all_loss=all_loss, metric=metric)

        return pred


@R.register("tasks.MultiGraphPreTraining")
class MultiGraphPreTraining(tasks.KnowledgeGraphCompletion, core.Configurable):
    
    def __init__(self, model, rel_models, criterion="bce", metric=("mr", "mrr", "hits@1", "hits@3", "hits@10"),
                 num_negative=128, margin=6, adversarial_temperature=0, strict_negative=True, filtered_ranking=True,
                 fact_ratio=None, sample_weight=True,
                 metric_per_rel=False):
        super(MultiGraphPreTraining, self).__init__(model, 
                 criterion=criterion, metric=metric,
                 num_negative=num_negative, margin=margin, 
                 adversarial_temperature=adversarial_temperature, 
                 strict_negative=strict_negative, filtered_ranking=filtered_ranking,
                 fact_ratio=fact_ratio, sample_weight=sample_weight)
        
        self.rel_models = rel_models
        self.metric_per_rel = metric_per_rel

    def preprocess(self, train_set, valid_set, test_set):
        dataset = train_set.dataset
        self.num_entity = dataset.num_entity
        self.num_relation = dataset.num_relation
        
        # Create graph, fact graph, and relation graph buffers for each graph in the dataset
        for i, graph in enumerate(dataset.graphs):
            fact_mask = torch.ones(graph.num_triplet, dtype=torch.bool)
            fact_mask[graph.num_samples[0]:] = 0
            #fact_mask[test_set.indices] = 0

            fact_graph = graph.graph.edge_mask(fact_mask)
            rel_graph = (self.rel_models[0]).construct_relation_graph(fact_graph)

            self.register_buffer(f"graph_{i}", graph.graph)
            self.register_buffer(f"fact_graph_{i}", fact_graph)
            self.register_buffer(f"rel_graph_{i}", rel_graph)

        self.num_graphs = len(dataset.graphs)

        return train_set, valid_set, test_set
    
    # our batch now has a graph id, so the only change is in getting pos_h/r/t index from batch[0]
    def forward(self, batch, all_loss=None, metric=None):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)
        pos_h_index, pos_t_index, pos_r_index = batch[0].t()

        for criterion, weight in self.criterion.items():
            if criterion == "bce":
                target = torch.zeros_like(pred)
                target[:, 0] = 1
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

                neg_weight = torch.ones_like(pred)
                if self.adversarial_temperature > 0:
                    with torch.no_grad():
                        neg_weight[:, 1:] = F.softmax(pred[:, 1:] / self.adversarial_temperature, dim=-1)
                else:
                    neg_weight[:, 1:] = 1 / self.num_negative
                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
            elif criterion == "ce":
                target = torch.zeros(len(pred), dtype=torch.long, device=self.device)
                loss = F.cross_entropy(pred, target, reduction="none")
            elif criterion == "ranking":
                positive = pred[:, :1]
                negative = pred[:, 1:]
                target = torch.ones_like(negative)
                loss = F.margin_ranking_loss(positive, negative, target, margin=self.margin)
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)

            if self.sample_weight:
                sample_weight = self.degree_hr[pos_h_index, pos_r_index] * self.degree_tr[pos_t_index, pos_r_index]
                sample_weight = 1 / sample_weight.float().sqrt()
                loss = (loss * sample_weight).sum() / sample_weight.sum()
            else:
                loss = loss.mean()

            name = tasks._get_criterion_name(criterion)
            metric[name] = loss
            all_loss += loss * weight

        return all_loss, metric
    
    def predict(self, batch, all_loss=None, metric=None):
        # assume the batch contains ID of one of the saved graphs
        pos_h_index, pos_t_index, pos_r_index = batch[0].t()
        graph_id = batch[1]
        batch_size = len(batch[0])

        # get here the fact graph and its relation graph
        graph = getattr(self, f"graph_{graph_id}")
        fact_graph = getattr(self, f"fact_graph_{graph_id}")
        rel_graph = getattr(self, f"rel_graph_{graph_id}")

        # get repr. for relations, assume only 1 rel model
        rel_inputs = []
        for i in range(len(self.rel_models)):
            if self.rel_models[0].__class__.__name__ != "RelNBFNet":
                rel_input = self.rel_models[i](rel_graph, None, all_loss=all_loss, metric=metric)["node_feature"]
            else:
                rel_input = self.rel_models[i](rel_graph, None, pos_r_index, all_loss=all_loss, metric=metric)["node_feature"]
            rel_inputs.append(rel_input)

        # the same as the Adapted class but the fact_graph is now extracted from self based on the graph id
        if all_loss is None:
            # test
            all_index = torch.arange(graph.num_node, device=self.device)
            t_preds = []
            h_preds = []
            for neg_index in all_index.split(self.num_negative):
                r_index = pos_r_index.unsqueeze(-1).expand(-1, len(neg_index))
                h_index, t_index = torch.meshgrid(pos_h_index, neg_index)
                t_pred = self.model(fact_graph, rel_inputs, h_index, t_index, r_index, all_loss=all_loss, metric=metric)
                t_preds.append(t_pred)
            t_pred = torch.cat(t_preds, dim=-1)
            for neg_index in all_index.split(self.num_negative):
                r_index = pos_r_index.unsqueeze(-1).expand(-1, len(neg_index))
                t_index, h_index = torch.meshgrid(pos_t_index, neg_index)
                h_pred = self.model(fact_graph, rel_inputs, h_index, t_index, r_index, all_loss=all_loss, metric=metric)
                h_preds.append(h_pred)
            h_pred = torch.cat(h_preds, dim=-1)
            pred = torch.stack([t_pred, h_pred], dim=1)
            # in case of GPU OOM
            pred = pred.cpu()
        else:
            # train
            if self.strict_negative:
                neg_index = self._strict_negative(pos_h_index, pos_t_index, pos_r_index, fact_graph)
            else:
                neg_index = torch.randint(graph.num_node, (batch_size, self.num_negative), device=self.device)
            h_index = pos_h_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            t_index = pos_t_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            r_index = pos_r_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            t_index[:batch_size // 2, 1:] = neg_index[:batch_size // 2]
            h_index[batch_size // 2:, 1:] = neg_index[batch_size // 2:]
            pred = self.model(fact_graph, rel_inputs, h_index, t_index, r_index, all_loss=all_loss, metric=metric)

        return pred
    
    # the same target but with the graph id and custom evaluation graph
    def target(self, batch):
        # test target
        batch, graph_id = batch
        graph = getattr(self, f"graph_{graph_id}")
        batch_size = len(batch)
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        any = -torch.ones_like(pos_h_index)

        pattern = torch.stack([pos_h_index, any, pos_r_index], dim=-1)
        edge_index, num_t_truth = graph.match(pattern)
        t_truth_index = graph.edge_list[edge_index, 1]
        pos_index = torch.repeat_interleave(num_t_truth)
        t_mask = torch.ones(batch_size, graph.num_node, dtype=torch.bool, device=self.device)
        t_mask[pos_index, t_truth_index] = 0

        pattern = torch.stack([any, pos_t_index, pos_r_index], dim=-1)
        edge_index, num_h_truth = graph.match(pattern)
        h_truth_index = graph.edge_list[edge_index, 0]
        pos_index = torch.repeat_interleave(num_h_truth)
        h_mask = torch.ones(batch_size, graph.num_node, dtype=torch.bool, device=self.device)
        h_mask[pos_index, h_truth_index] = 0

        mask = torch.stack([t_mask, h_mask], dim=1)
        target = torch.stack([pos_t_index, pos_h_index], dim=1)

        # in case of GPU OOM
        return mask.cpu(), target.cpu()
    
    # the same as in the parent class except that :
    # (1) we also send the fact graph in which we look for the negatives
    # (2) num nodes is taken from the fact graph
    @torch.no_grad()
    def _strict_negative(self, pos_h_index, pos_t_index, pos_r_index, fact_graph):
        batch_size = len(pos_h_index)
        any = -torch.ones_like(pos_h_index)

        pattern = torch.stack([pos_h_index, any, pos_r_index], dim=-1)
        pattern = pattern[:batch_size // 2]
        edge_index, num_t_truth = fact_graph.match(pattern)
        t_truth_index = fact_graph.edge_list[edge_index, 1]
        pos_index = torch.repeat_interleave(num_t_truth)
        t_mask = torch.ones(len(pattern), fact_graph.num_node, dtype=torch.bool, device=self.device)
        t_mask[pos_index, t_truth_index] = 0
        neg_t_candidate = t_mask.nonzero()[:, 1]
        num_t_candidate = t_mask.sum(dim=-1)
        neg_t_index = functional.variadic_sample(neg_t_candidate, num_t_candidate, self.num_negative)

        pattern = torch.stack([any, pos_t_index, pos_r_index], dim=-1)
        pattern = pattern[batch_size // 2:]
        edge_index, num_h_truth = fact_graph.match(pattern)
        h_truth_index = fact_graph.edge_list[edge_index, 0]
        pos_index = torch.repeat_interleave(num_h_truth)
        h_mask = torch.ones(len(pattern), fact_graph.num_node, dtype=torch.bool, device=self.device)
        h_mask[pos_index, h_truth_index] = 0
        neg_h_candidate = h_mask.nonzero()[:, 1]
        num_h_candidate = h_mask.sum(dim=-1)
        neg_h_index = functional.variadic_sample(neg_h_candidate, num_h_candidate, self.num_negative)

        neg_index = torch.cat([neg_t_index, neg_h_index])

        return neg_index

    def predict_and_target(self, batch, all_loss=None, metric=None):
        pred, target = super(MultiGraphPreTraining, self).predict_and_target(batch, all_loss, metric)
        if self.graph.num_node > 1e6:
            ranking = self.get_ranking(pred, target)
            return ranking, ranking
        else:
            return pred, target

    def get_ranking(self, pred, target):
        mask, target = target
        pos_pred = pred.gather(-1, target.unsqueeze(-1))
        if self.filtered_ranking:
            ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
        else:
            ranking = torch.sum(pos_pred <= pred, dim=-1) + 1

        return ranking

    def evaluate(self, pred, target):
        if not isinstance(target, torch.Tensor):
            ranking = self.get_ranking(pred, target)
        else:
            ranking = pred

        metric = {}
        for _metric in self.metric:
            # e.g. mrr-tail means mrr for tail prediction
            if "-" in _metric:
                _metric, direction = _metric.split("-")
                if direction == "head":
                    _ranking = ranking.select(1, 1)
                elif direction == "tail":
                    _ranking = ranking.select(1, 0)
                else:
                    raise ValueError("Unknown direction `%s`" % direction)
            else:
                _ranking = ranking
            if _metric == "mr":
                score = _ranking.float().mean()
            elif _metric == "mrr":
                score = (1 / _ranking.float()).mean()
            elif _metric.startswith("hits@"):
                threshold = int(_metric[5:])
                score = (_ranking <= threshold).float().mean()
            else:
                raise ValueError("Unknown metric `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            metric[name] = score

        return metric
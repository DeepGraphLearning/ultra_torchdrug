import os
import sys
import random

import numpy as np
import argparse
import time
import csv

import torch
from torch.utils import data as torch_data
import torch_geometric.data
from torchdrug import core, utils
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra import layer, dataset, rel_model, model, task, util, engine

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from run_full import train_and_validate
util.setup_debug_hook()

default_finetuning_config = {
    # graph: (num_epochs, batches_per_epoch), null means all triples in train set
    # transductive datasets (17)
    # standard ones (10)
    "CoDExSmall": (1, 4000),
    "CoDExMedium": (1, 4000),
    "CoDExLarge": (1, 2000),
    "FB15k237": (1, 'null'),
    "WN18RR": (1, 'null'),
    "YAGO310": (1, 2000),
    "DBpedia100k": (1, 1000),
    "AristoV4": (1, 2000),
    "ConceptNet100k": (1, 2000),
    "ATOMIC": (1, 200),
    # tail-only datasets (2)
    "NELL995": (1, 'null'),  # not implemented yet
    "Hetionet": (1, 4000),
    # sparse datasets (5)
    "WDsinger": (3, 'null'),
    "FB15k237_10": (1, 'null'),
    "FB15k237_20": (1, 'null'),
    "FB15k237_50": (1, 1000),
    "NELL23k": (3, 'null'),
    # inductive datasets (42)
    # GraIL datasets (12)
    "FB15k237Inductive": (1, 'null'),    # for all 4 datasets
    "WN18RRInductive": (1, 'null'),      # for all 4 datasets
    "NELLInductive": (3, 'null'),        # for all 4 datasets
    # ILPC (2)
    "ILPC2022SmallInductive": (3, 'null'),
    "ILPC2022LargeInductive": (1, 1000),
    # Ingram datasets (13)
    "NLIngram": (3, 'null'),  # for all 5 datasets
    "FBIngram": (3, 'null'),  # for all 4 datasets
    "WKIngram": (3, 'null'),  # for all 4 datasets
    # MTDEA datasets (10)
    "WikiTopicsMT1": (3, 'null'),  # for all 2 test datasets
    "WikiTopicsMT2": (3, 'null'),  # for all 2 test datasets
    "WikiTopicsMT3": (3, 'null'),  # for all 2 test datasets
    "WikiTopicsMT4": (3, 'null'),  # for all 2 test datasets
    "Metafam": (3, 'null'),
    "FBNELL": (3, 'null'),
    # Hamaguchi datasets (5)
    "HamaguchiBM": (1, 100)  # for all 5 datasets
}

default_train_config = {
    # graph: (num_epochs, batches_per_epoch), null means all triples in train set
    # transductive datasets (17)
    # standard ones (10)
    "CoDExSmall": (10, 1000),
    "CoDExMedium": (10, 1000),
    "CoDExLarge": (10, 1000),
    "FB15k237": (10, 1000),
    "WN18RR": (10, 1000),
    "YAGO310": (10, 2000),
    "DBpedia100k": (10, 1000),
    "AristoV4": (10, 1000),
    "ConceptNet100k": (10, 1000),
    "ATOMIC": (10, 1000),
    # tail-only datasets (2)
    "NELL995": (10, 1000),  # not implemented yet
    "Hetionet": (10, 1000),
    # sparse datasets (5)
    "WDsinger": (10, 1000),
    "FB15k237_10": (10, 1000),
    "FB15k237_20": (10, 1000),
    "FB15k237_50": (10, 1000),
    "NELL23k": (10, 1000),
    # inductive datasets (42)
    # GraIL datasets (12)
    "FB15k237Inductive": (10, 'null'),    # for all 4 datasets
    "WN18RRInductive": (10, 'null'),      # for all 4 datasets
    "NELLInductive": (10, 'null'),        # for all 4 datasets
    # ILPC (2)
    "ILPC2022SmallInductive": (10, 'null'),
    "ILPC2022LargeInductive": (10, 1000),
    # Ingram datasets (13)
    "NLIngram": (10, 'null'),  # for all 5 datasets
    "FBIngram": (10, 'null'),  # for all 4 datasets
    "WKIngram": (10, 'null'),  # for all 4 datasets
    # MTDEA datasets (10)
    "WikiTopicsMT1": (10, 'null'),  # for all 2 test datasets
    "WikiTopicsMT2": (10, 'null'),  # for all 2 test datasets
    "WikiTopicsMT3": (10, 'null'),  # for all 2 test datasets
    "WikiTopicsMT4": (10, 'null'),  # for all 2 test datasets
    "Metafam": (10, 'null'),
    "FBNELL": (10, 'null'),
    # Hamaguchi datasets (5)
    "HamaguchiBM": (10, 1000)  # for all 5 datasets
}

def build_solver(cfg, dataset):
    if torch.cuda.is_available():
        torch.cuda.set_device(comm.get_rank())
        torch.cuda.empty_cache()
    
    train_set, valid_set, test_set = dataset.split()
    if comm.get_rank() == 0:
        logger.warning(dataset)
        logger.warning("#train: %d, #valid: %d, #test: %d" % (len(train_set), len(valid_set), len(test_set)))

    if "fast_test" in cfg:
        if comm.get_rank() == 0:
            logger.warning("Quick test mode on. Only evaluate on %d samples for valid / test." % cfg.fast_test)
        g = torch.Generator()
        g.manual_seed(1024)
        if cfg.task["class"] != "MultiGraphPreTraining":
            valid_set = torch_data.random_split(valid_set, [cfg.fast_test, len(valid_set) - cfg.fast_test], generator=g)[0]
            test_set = torch_data.random_split(test_set, [cfg.fast_test, len(test_set) - cfg.fast_test], generator=g)[0]
        else:
            valid_set = {graph_name: (vals[0], torch.arange(vals[1][0], vals[1][1])[torch.randperm(vals[1][1] - vals[1][0])][:cfg.fast_test]) for graph_name, vals in valid_set.items()}
            test_set = {graph_name: (vals[0], torch.arange(vals[1][0], vals[1][1])[torch.randperm(vals[1][1] - vals[1][0])][:cfg.fast_test]) for graph_name, vals in test_set.items()}
    if hasattr(dataset, "num_relation"):
        cfg.task.model.num_relation = dataset.num_relation.item()
        if hasattr(cfg.task, "rel_models"):
            cfg.task.rel_models.num_relation = dataset.num_relation.item() * 2
    

    task = core.Configurable.load_config_dict(cfg.task)
    cfg.optimizer.params = task.parameters()
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)
    solver_cls = core.Engine if cfg.task["class"] != "MultiGraphPreTraining" else engine.MultiGraphEngine
    solver = solver_cls(task, train_set, valid_set, test_set, optimizer, scheduler=None, **cfg.engine)
    fix_reasoner = cfg.get("fix_reasoner", False)
    if fix_reasoner:
        assert cfg.task["class"] == "KnowledgeGraphCompletion" or cfg.task["class"] == "KnowledgeGraphCompletionAdapted"
    if "checkpoint" in cfg:
        util.safe_load(solver, cfg.checkpoint, fix_reasoner=fix_reasoner)  # clean load without graph objects
    
    return solver

def set_seed(seed):
    random.seed(seed + comm.get_rank())
    np.random.seed(seed + comm.get_rank())
    torch.manual_seed(seed + comm.get_rank())
    torch.cuda.manual_seed(seed + comm.get_rank())
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test(cfg, solver):
    solver.model.split = "valid"
    solver.evaluate("valid")
    if hasattr(cfg, "no_test"):
        exit(0)
        return
    solver.model.split = "test"
    metrics = solver.evaluate("test")
    return metrics

if __name__ == "__main__":
    
    seeds = [1024, 42, 1337, 512, 256]

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-d", "--datasets", help="target datasets", default='FB15k237Inductive:v1,NELLInductive:v4', type=str, required=True)
    parser.add_argument("-reps", "--repeats", help="number of times to repeat each exp", default=1, type=int)
    parser.add_argument("-ft", "--finetune", help="finetune the checkpoint on the specified datasets", action='store_true')
    parser.add_argument("-tr", "--train", help="train the model from scratch", action='store_true')
    args, unparsed = parser.parse_known_args()
   
    datasets = args.datasets.split(",")
    path = os.path.dirname(os.path.expanduser(__file__))
    results_file = os.path.join(path, f"ultra_results_{time.strftime('%Y-%m-%d-%H-%M-%S')}.csv")

   
    for graph in datasets:
        ds, version = graph.split(":") if ":" in graph else (graph, None)
        for i in range(args.repeats):
            seed = seeds[i] if i < len(seeds) else random.randint(0, 10000)
            print(f"Running on {graph}, iteration {i+1} / {args.repeats}, seed: {seed}")

            # get dynamic arguments defined in the config file
            vars = util.detect_variables(args.config)
            parser = argparse.ArgumentParser()
            for var in vars:
                parser.add_argument("--%s" % var)
            vars = parser.parse_known_args(unparsed)[0]
            vars = {k: utils.literal_eval(v) for k, v in vars._get_kwargs()}
            
            if args.finetune:
                epochs, batch_per_epoch = default_finetuning_config[ds] 
            elif args.train:
                epochs, batch_per_epoch = default_train_config[ds] 
            else:
                epochs, batch_per_epoch = 0, 'null'
            vars['epochs'] = epochs
            vars['bpe'] = batch_per_epoch
            vars['dataset'] = ds
            if version is not None:
                vars['version'] = version
            cfg = util.load_config(args.config, context=vars)[0]

            root_dir = os.path.expanduser(cfg.output_dir) # resetting the path to avoid inf nesting
            os.chdir(root_dir)
            working_dir = util.create_working_directory(cfg)
            set_seed(seed)

            logger = util.get_root_logger()

            data = core.Configurable.load_config_dict(cfg.dataset)
            solver = build_solver(cfg, data)

            with torch.autograd.set_detect_anomaly(True):
                train_and_validate(cfg, solver)
                metrics = test(cfg, solver)

            metrics = {k:v.item() for k,v in metrics.items()}
            metrics['dataset'] = graph
            # write to the log file
            with open(results_file, "a", newline='') as csv_file:
                fieldnames = ['dataset']+list(metrics.keys())[:-1]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=',')
                if csv_file.tell() == 0:
                    writer.writeheader()
                writer.writerow(metrics)
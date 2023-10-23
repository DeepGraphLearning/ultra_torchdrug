import os
import sys
import math
import pprint
import random

import numpy as np

import torch
from torch.utils import data as torch_data
import torch_geometric.data

from torchdrug import core
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra import layer, dataset, rel_model, model, task, util, engine

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


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

def train_and_validate(cfg, solver):
    if cfg.train.num_epoch == 0:
        return

    step = math.ceil(cfg.train.num_epoch / 10)
    best_result = float("-inf")
    best_epoch = -1

    for i in range(0, cfg.train.num_epoch, step):
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i)
        if cfg.debug:
            kwargs["epoch_id"] = i
        if hasattr(cfg.train, "clip_grad") and cfg.debug:
            kwargs["clip_grad"] = cfg.train.clip_grad
        solver.model.split = "train"
        solver.train(**kwargs)
        util.clean_save(solver, "model_epoch_%d.pth" % solver.epoch)  # clean save without graph objects
        solver.model.split = "valid"
        metric = solver.evaluate("valid")
        result = metric[cfg.metric]
        if result > best_result:
            best_result = result
            best_epoch = solver.epoch

    checkpoint = "model_epoch_%d.pth" % best_epoch
    util.safe_load(solver, checkpoint)

    return solver


def test(cfg, solver):
    solver.model.split = "valid"
    solver.evaluate("valid")
    if hasattr(cfg, "no_test"):
        exit(0)
        return
    solver.model.split = "test"
    solver.evaluate("test")

def set_seed(args):
    seed = args.seed
    random.seed(seed + comm.get_rank())
    np.random.seed(seed + comm.get_rank())
    torch.manual_seed(seed + comm.get_rank())
    torch.cuda.manual_seed(seed + comm.get_rank())
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    
    args, vars = util.parse_args()

    cfg = util.load_config(args.config, context=vars)[0]
    working_dir = util.create_working_directory(cfg)
    set_seed(args)

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver = build_solver(cfg, dataset)

    with torch.autograd.set_detect_anomaly(True):
        train_and_validate(cfg, solver)
        test(cfg, solver)

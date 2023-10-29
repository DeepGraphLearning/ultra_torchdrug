import os
import sys
import time
import logging
import argparse

import yaml
import easydict
import jinja2
from jinja2 import meta
from copy import deepcopy

import torch
from torch import distributed as dist

from torchdrug import core, utils
from torchdrug.utils import comm
from torch_scatter import scatter_add
from torch_geometric.utils import remove_self_loops


logger = logging.getLogger(__file__)

def meshgrid(dict):
    if len(dict) == 0:
        yield {}
        return

    key = next(iter(dict))
    values = dict[key]
    sub_dict = dict.copy()
    sub_dict.pop(key)

    if not isinstance(values, list):
        values = [values]
    for value in values:
        for result in meshgrid(sub_dict):
            result[key] = value
            yield result


def load_config(cfg_file, context=None):
    with open(cfg_file, "r") as fin:
        raw_text = fin.read()
        
    if "---" in raw_text:
        configs = []
        grid, template = raw_text.split("---")
        grid = yaml.safe_load(grid)
        template = jinja2.Template(template)
        for hyperparam in meshgrid(grid):
            if context is not None:
                hyperparam = hyperparam + context
            instance = template.render(hyperparam)
            config = easydict.EasyDict(yaml.safe_load(instance))
            configs.append(config)
    else:
        if context is not None:
            template = jinja2.Template(raw_text)
            instance = template.render(context)
            configs = [easydict.EasyDict(yaml.safe_load(instance))]
        else:
            configs = [easydict.EasyDict(yaml.safe_load(raw_text))]

    #import ipdb
    #ipdb.set_trace()
    #template = jinja2.Template(raw_text)
    #instance = template.render(context)
    #cfg = yaml.safe_load(instance)
    #cfg = easydict.EasyDict(cfg)
    return configs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)

    args, unparsed = parser.parse_known_args()
    # get dynamic arguments defined in the config file
    vars = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, required=True)
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: utils.literal_eval(v) for k, v in vars._get_kwargs()}

    return args, vars


def get_root_logger(file=True):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

    if file:
        handler = logging.FileHandler("log.txt")
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def create_working_directory(cfg):
    file_name = "working_dir.tmp" #% os.environ.get("SLURM_JOB_ID", "")
    world_size = comm.get_world_size()
    if world_size > 1 and not dist.is_initialized():
        comm.init_process_group("nccl", init_method="env://")

    working_dir = os.path.join(os.path.expanduser(cfg.output_dir),
                               cfg.task["class"], cfg.dataset["class"], cfg.task.model["class"],
                               time.strftime("%Y-%m-%d-%H-%M-%S"))

    # synchronize working directory
    if comm.get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(working_dir)
        os.makedirs(working_dir)
    comm.synchronize()
    if comm.get_rank() != 0:
        with open(file_name, "r") as fin:
            working_dir = fin.read()
    comm.synchronize()
    if comm.get_rank() == 0:
        os.remove(file_name)

    os.chdir(working_dir)
    return working_dir


def detect_variables(cfg_file):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    ast = env.parse(raw)
    vars = meta.find_undeclared_variables(ast)
    return vars

class DebugHook:
    instance = None

    def __call__(self, *args, **kwargs):
        if comm.get_rank() > 0:
            while True:
                pass

        if self.instance is None:
            from IPython.core import ultratb
            self.instance = ultratb.FormattedTB(mode="Plain", color_scheme="Linux", call_pdb=1)
        return self.instance(*args, **kwargs)


def setup_debug_hook():
    sys.excepthook = DebugHook()

def random_walk_se(graph, ksteps, return_all=False, remove_loops=False):
    """Compute RWSE - diagonals of random walk matrices up to k-th degree
        Optionally, return all random walk matrices to be used as edge features
    """
    num_nodes = graph.num_node

    # deduplicate edges
    edge_list = graph.edge_list.unique(dim=0).T
    if remove_loops:
        # remove self-loops
        edge_list = remove_self_loops(edge_index=edge_list)[0]

    src, dst = edge_list
    edge_weight = torch.ones(len(src), device=graph.edge_list.device)
    adj = torch.sparse_coo_tensor(
        #graph.edge_list[:, [0, 1]].T,
        edge_list,
        edge_weight,
        (num_nodes, num_nodes)
    )

    deg = scatter_add(edge_weight, src, dim=0, dim_size=num_nodes) # out degrees
    deg_inv = deg.pow(-1)  # D ^ -1
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    # torch.sparse.spdiags is available from torch 1.13, for backward compatibility:
    deg_inv = torch.sparse_coo_tensor(
        indices=torch.arange(deg_inv.shape[0], device=deg_inv.device).unsqueeze(dim=0).repeat(2,1),
        values=deg_inv
    )
    #deg_inv2 = torch.sparse.spdiags(deg_inv, torch.tensor([0]), (num_nodes, num_nodes))

    P = torch.sparse.mm(deg_inv, adj)  # D^-1 * A normalization, rows sum to 1

    rws, rrpes = [], []
    Pk = P.clone().detach()
    # for removed self-loops, the first dimension will always be zeros, so do ksteps+1
    for k in range(ksteps+1 if remove_loops else ksteps):
        rws.append(safe_diagonal(Pk))
        if return_all:
            rrpes.append(Pk.to_dense())
        Pk = torch.sparse.mm(Pk, P)

    rw_landing = torch.stack(rws).transpose(0,1)  # num_node x dim
    rrpe = torch.stack(rrpes).permute(1,2,0) if return_all else None  # num_node x num_node x dim
    if remove_loops:
        # extract the 0th dim because they are all zeros anyways
        rw_landing = rw_landing[:, 1:]
        rrpe = rrpe[..., 1:] if rrpe is not None else None
    return rw_landing if not return_all else (rw_landing, rrpe)

def safe_diagonal(matrix):
    """
    Extract diagonal from a potentially sparse matrix.
    .. note ::
        this is a work-around as long as :func:`torch.diagonal` does not work for sparse tensors
    :param matrix: shape: `(n, n)`
        the matrix
    :return: shape: `(n,)`
        the diagonal values.
    """
    if not matrix.is_sparse:
        return torch.diagonal(matrix)

    # # convert to COO, if necessary <- comment that as we only work with COO
    # if matrix.is_sparse_csr:
    #     matrix = matrix.to_sparse_coo()

    n = matrix.shape[0]
    # we need to use indices here, since there may be zero diagonal entries
    indices = matrix._indices()
    mask = indices[0] == indices[1]
    diagonal_values = matrix._values()[mask]
    diagonal_indices = indices[0][mask]

    return torch.zeros(n, device=matrix.device).scatter_add(dim=0, index=diagonal_indices, src=diagonal_values)


def safe_load(solver, checkpoint, fix_reasoner=False, drop_optimizer=True):

    # safe load of the checkpoint without loading possible non-tensor object like saved graphs
    if comm.get_rank() == 0:
        logger.warning("Load checkpoint from %s" % checkpoint)
    checkpoint = os.path.expanduser(checkpoint)
    state = torch.load(checkpoint, map_location=solver.device)

    keys_to_delete = ["fact_graph", "train_graph", "valid_graph", "test_graph", 
                        "train_rel_graph", "valid_rel_graph", "test_rel_graph",
                        "graph", "inductive_graph", "train_rel_graphs", 
                        "valid_rel_graphs", "test_rel_graphs", "rel_graphs"]
    for key in keys_to_delete:
        if key in list(state["model"].keys()) and not torch.is_tensor(state["model"][key]):
            state["model"].pop(key)
    
    if fix_reasoner:
        keys_to_ignore = []
        for key in list(state["model"].keys()):
            if "relation.weight" in key or "relation_projection" in key or "relation_linear" in key or "query.weight" in key:
                keys_to_ignore.append(key)
        current_weights = solver.model.state_dict()
        #ipdb.set_trace()
        for key in keys_to_ignore:
            state["model"].pop(key)
            if key in current_weights:
                state["model"][key] = current_weights[key]
        
    solver.model.load_state_dict(state["model"], strict=False)

    # don't load the optimizer state for fresh fine-tuning
    if not drop_optimizer:
        try:
            if not fix_reasoner:
                solver.optimizer.load_state_dict(state["optimizer"])
        except:
            print("\n\n\n\nwarning: loaded state dict has a different number of parameter groups\n\n\n\n")

        for state in solver.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(solver.device)

    comm.synchronize()

def clean_save(solver, checkpoint):

    # exclude non-tensor objects like saved graphs from the checkpoint
    if comm.get_rank() == 0:
        logger.warning("Save checkpoint to %s" % checkpoint)
    checkpoint = os.path.expanduser(checkpoint)

    # the solver itself is used for training even after saving
    # so we do all manipulations with the copy of the state dict object and save the copy
    
    # complexes object does not allow detach() in torch=1.13.1
    if hasattr(solver.model._buffers, "train_rel_graphs"):
        train_rel_graph = solver.model._buffers['train_rel_graphs']
        valid_rel_graph = solver.model._buffers['valid_rel_graphs']
        test_rel_graph = solver.model._buffers['test_rel_graphs']
        delattr(solver.model, "train_rel_graphs")
        delattr(solver.model, "valid_rel_graphs")
        delattr(solver.model, "test_rel_graphs")

        model_to_save = deepcopy(solver.model.state_dict())

        solver.model.register_buffer("train_rel_graphs", train_rel_graph)
        solver.model.register_buffer("valid_rel_graphs", valid_rel_graph)
        solver.model.register_buffer("test_rel_graphs", test_rel_graph)
    elif hasattr(solver.model._buffers, "rel_graphs"):
        rel_graphs = solver.model._buffers['rel_graphs']
        delattr(solver.model, "rel_graphs")
        model_to_save = deepcopy(solver.model.state_dict())
        solver.model.register_buffer("rel_graphs", rel_graphs)
    else:
        model_to_save = deepcopy(solver.model.state_dict())

    keys_to_delete = ["fact_graph", "train_graph", "valid_graph", "test_graph", 
                        "train_rel_graph", "valid_rel_graph", "test_rel_graph",
                        "graph", "inductive_graph", "train_rel_graphs", 
                        "valid_rel_graphs", "test_rel_graphs"]
    for key in keys_to_delete:
        if key in list(model_to_save.keys()) and not torch.is_tensor(model_to_save[key]):
            model_to_save.pop(key)

    if solver.rank == 0:
        state = {
            "model": model_to_save,
            "optimizer": solver.optimizer.state_dict()
        }
        torch.save(state, checkpoint)

    comm.synchronize()

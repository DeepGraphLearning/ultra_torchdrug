from collections import Sequence
from decorator import decorator

import torch
from torch import nn
from torch_scatter import scatter_add, scatter_min

import torchdrug as td
from torchdrug import data
from torchdrug import core, layers, utils
from torchdrug.layers import functional
from torchdrug.core import Registry as R
from torchdrug.utils import comm

from . import layer, functional
from .util import random_walk_se


@decorator # signature-changing decorator
def _reset_input(forward_func, self, graph, input, *args, **kwargs):
    assert input is None

    #print("func: %s was called" % forward_func.__name__)
    #kwstr = ', '.join('%r: %r' % (k, kwargs[k]) for k in sorted(kwargs))
    #print("args: ", forward_func, self, graph, input, args, kwstr)
    input = 0
    
    keys = self.input_type.split("__")
    for k in keys:
        if k == "glorot":
            input += nn.init.xavier_uniform_(torch.zeros(graph.num_node, self.input_dim, device=self.device))
        if k == "ones":
            input += torch.ones(graph.num_node, self.input_dim, device=self.device)
        if k == "zeros":
            input += torch.zeros(graph.num_node, self.input_dim, device=self.device)
        if k == "embedding":
            input += self.rel_embedding.weight
        if k == "degree_encoding":
            input += self.deg_embedding(graph.degree_out)
        if k == "random_walk_encoding":
            bucketized_rw = (graph.node_feature * 1000).int()[:, :8]
            bucketized_rw = torch.min(bucketized_rw, torch.tensor(self.num_random_walk_bucket, device=self.device, dtype=torch.int))
            input += self.random_walk_embedding(bucketized_rw).mean(dim=1)
            #ipdb.set_trace()
        if k != "glorot" and k != "ones" and k != "zeros" and k != "embedding":
            input += graph.node_feature
        #if comm.get_rank() == 0:
        #    print("use node features ...")
        #ipdb.set_trace()
    
    return forward_func(self, graph, input, *args, **kwargs)


class RelationModel(nn.Module, core.Configurable):

    num_degrees = 1000
    num_distance = 2000
    num_random_walk_bucket = 40

    def __init__(self, input_type="glorot", num_bins=1, 
        mine_complexes=False, num_relation=None, output_layer_norm=False, remove_self_loops=False,
        multirelational=False, ablation_etypes=False, **kwargs):
        """
        Include hyper-parameters for relation graph construction here
        """
        super(RelationModel, self).__init__()

        self.input_type = input_type
        self.num_bins = num_bins
        self.mine_complexes = mine_complexes
        self.num_relation = num_relation
        self.output_layer_norm = output_layer_norm
        self.remove_self_loops = remove_self_loops
        self.save_multirelational = multirelational
        self.ablation_etypes = ablation_etypes
    
    def _get_shortest_distance(self, graph, num_iters=100):

        dist = torch.ones(graph.num_node, graph.num_node, dtype=torch.int32, device=self.device) * graph.num_node
        for i, start in enumerate(range(graph.num_node)):
            dist[start, i] = 0
        for i in range(num_iters):

            node_in, node_out = graph.edge_list[:, :2].t()
            message = dist[node_in] + 1
            update, _ = scatter_min(message, node_out, dim=0, dim_size=graph.num_node)

            dist[node_out] = torch.min(dist[node_out], update[node_out])
        return dist
    
    def construct_relation_graph(self, graph):
        graph = graph.undirected(add_inverse=True)
        device = graph.device
        
        Eh = graph.edge_list[:, [0, 2]].unique(dim=0)
        Dh = scatter_add(torch.ones_like(Eh[:, 1]), Eh[:, 0])
        assert not (Dh[Eh[:, 0]] == 0).any()

        EhT = torch.sparse_coo_tensor(torch.flip(Eh, dims=[1]).T, 
                torch.ones(Eh.shape[0], device=device) / Dh[Eh[:, 0]], 
                (graph.num_relation, graph.num_node)
            )
        Eh = torch.sparse_coo_tensor(Eh.T, 
                torch.ones(Eh.shape[0], device=device), 
                (graph.num_node, graph.num_relation)
            )

        Et = graph.edge_list[:, [1, 2]].unique(dim=0)
        Dt = scatter_add(torch.ones_like(Et[:, 1]), Et[:, 0])
        assert not (Dt[Et[:, 0]] == 0).any()

        EtT = torch.sparse_coo_tensor(torch.flip(Et, dims=[1]).T, 
                torch.ones(Et.shape[0], device=device) / Dt[Et[:, 0]], 
                (graph.num_relation, graph.num_node)
            )
        Et = torch.sparse_coo_tensor(Et.T, 
                torch.ones(Et.shape[0], device=device), 
                (graph.num_node, graph.num_relation)
            )

        Ahh = torch.sparse.mm(EhT, Eh).coalesce()
        Att = torch.sparse.mm(EtT, Et).coalesce()
        Aht = torch.sparse.mm(EhT, Et).coalesce()
        Ath = torch.sparse.mm(EtT, Eh).coalesce()
        A = (Ahh + Att).coalesce()

        bin_indices = torch.div(Ahh.values().sort().indices * self.num_bins, Ahh.values().shape[0], rounding_mode='floor')
        hh_rel_graph = td.data.Graph(Ahh.indices().T, edge_weight=bin_indices, num_node=graph.num_relation)
        bin_indices = torch.div(Att.values().sort().indices * self.num_bins, Att.values().shape[0], rounding_mode='floor')
        tt_rel_graph = td.data.Graph(Att.indices().T, edge_weight=bin_indices, num_node=graph.num_relation)
        bin_indices = torch.div(Aht.values().sort().indices * self.num_bins, Aht.values().shape[0], rounding_mode='floor')
        ht_rel_graph = td.data.Graph(Aht.indices().T, edge_weight=bin_indices, num_node=graph.num_relation)
        bin_indices = torch.div(Ath.values().sort().indices * self.num_bins, Ath.values().shape[0], rounding_mode='floor')
        th_rel_graph = td.data.Graph(Ath.indices().T, edge_weight=bin_indices, num_node=graph.num_relation)
        bin_indices = torch.div(A.values().sort().indices * self.num_bins, A.values().shape[0], rounding_mode='floor')
        rel_graph = td.data.Graph(A.indices().T, edge_weight=bin_indices, num_node=graph.num_relation)

        if self.save_multirelational:
            hh_edges = torch.cat([Ahh.indices().T, torch.zeros(Ahh.indices().T.shape[0], 1, dtype=torch.long).fill_(0)], dim=1)
            tt_edges = torch.cat([Att.indices().T, torch.zeros(Att.indices().T.shape[0], 1, dtype=torch.long).fill_(1)], dim=1)
            ht_edges = torch.cat([Aht.indices().T, torch.zeros(Aht.indices().T.shape[0], 1, dtype=torch.long).fill_(2)], dim=1)
            th_edges = torch.cat([Ath.indices().T, torch.zeros(Ath.indices().T.shape[0], 1, dtype=torch.long).fill_(3)], dim=1)
            rel_graph = td.data.Graph(torch.cat([hh_edges, tt_edges, ht_edges, th_edges], dim=0), num_node=graph.num_relation, num_relation=4)

            if self.ablation_etypes:
                edges = (Ahh + Att +Aht + Ath).coalesce()
                rel_graph = td.data.Graph(edges.indices().T, num_node=graph.num_relation)
        

        if "random_walk" in self.input_type:
            assert self.input_dim is not None
            initial_features = random_walk_se(rel_graph, self.input_dim, remove_loops=self.remove_self_loops)
            
            with rel_graph.node():
                rel_graph.node_feature = initial_features

        elif "rrpe" in self.input_type:
            assert self.input_dim is not None
            initial_features, edge_features = random_walk_se(rel_graph, self.input_dim, return_all=True, remove_loops=self.remove_self_loops)
            
            with rel_graph.node():
                rel_graph.node_feature = initial_features
                rel_graph.edge_features = edge_features
        else:
            initial_features = torch.ones(rel_graph.num_node, self.input_dim, device=self.device)
        
        
        if "random_walk_inspect" in self.input_type:
            import ipdb
            ipdb.set_trace()
        
        if "embedding" in self.input_type:
            self.rel_embedding = nn.Embedding(self.num_relation, self.input_dim)
        if "degree_encoding" in self.input_type:
            self.deg_embedding = nn.Embedding(self.num_degrees, self.input_dim)
        if "distance_encoding" in self.input_type:
            self.dist_embedding = nn.Embedding(self.num_distance, self.input_dim)
        if "random_walk_encoding" in self.input_type:
            self.random_walk_embedding = nn.Embedding(self.num_random_walk_bucket + 1, self.input_dim)

        return rel_graph
    

    @_reset_input
    def forward(self, graph, input, all_loss=None, metric=None):
        
        hiddens = []
        layer_input = input

        for layer in self.layers:
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }
                

@R.register("models.RelationModelList")
class RelationModelList(nn.ModuleList, core.Configurable):

    def __init__(self, num_rel_models=1, num_relation=None, *args, **kwargs):
        super(RelationModelList, self).__init__()

        self.num_rel_models = num_rel_models
        self.num_relation = num_relation

        rel_model_cfg = kwargs['rel_model'].copy()
        rel_model_cfg.pop('class_str')
        for i in range(num_rel_models):
            rel_model = eval(kwargs['rel_model']['class_str'])(
                                **rel_model_cfg, num_relation=self.num_relation)
            self.append(rel_model)


# Custom NBFNet here without final feature->1 mlp and with query representations as ones
class CustomNBFNet(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dims, num_relation=None, symmetric=False,
                 message_func="distmult", aggregate_func="pna", short_cut=False, layer_norm=False, activation="relu",
                 concat_hidden=False, num_mlp_layer=2, dependent=False, remove_one_hop=False,
                 num_beam=10, path_topk=10,
                 separate_remove_one_hop=False):
        super(CustomNBFNet, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        if num_relation is None:
            double_relation = 1
            num_relation = 1
        else:
            num_relation = int(num_relation)
            double_relation = num_relation * 2
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.symmetric = symmetric
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.remove_one_hop = remove_one_hop
        self.num_beam = num_beam
        self.path_topk = path_topk

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layer.GeneralizedRelationalConvNBF(self.dims[i], self.dims[i + 1], num_relation, #double_relation,
                                                               self.dims[0], message_func, aggregate_func, layer_norm,
                                                               activation, dependent))

        feature_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1) + input_dim

        # self.query = nn.Embedding(double_relation, input_dim)
        
        self.mlp = layers.MLP(feature_dim, [feature_dim] * (num_mlp_layer - 1) + [hidden_dims[-1]])

        # self.dist_embed = nn.Embedding(10, input_dim)
       

    @utils.cached
    def bellmanford(self, graph, h_index, separate_grad=False):
        # query = self.query(r_index)
        query = torch.ones(h_index.shape[0], self.dims[0], device=h_index.device, dtype=torch.float)
        index = h_index#.unsqueeze(-1).expand_as(query)
        # important: do not label nodes uniquely per batch, initialize one graph with batch_size num of labeled nodes
        #boundary = torch.zeros(graph.num_node, *query.shape, device=self.device)
        boundary = torch.zeros(graph.num_node, query.shape[1], device=self.device)
        boundary[h_index] = 1.0
        #boundary.scatter_add_(0, index.unsqueeze(0), query.unsqueeze(0))

        with graph.graph():
            graph.query = query.unsqueeze(0)
        with graph.node():
            graph.boundary = boundary.unsqueeze(1)

        hiddens = []
        layer_input = boundary.unsqueeze(1)

        for layer in self.layers:
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        # node_query = query.expand(graph.num_node, -1, -1)
        # if self.concat_hidden:
        #     output = torch.cat(hiddens + [node_query], dim=-1)
        # else:
        #     output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": hiddens[-1].squeeze(1),
        }
    
    def as_relational_graph(self, graph, self_loop=False):
        # add self loop
        # convert homogeneous graphs to knowledge graphs with 1 relation
        edge_list = graph.edge_list
        edge_weight = graph.edge_weight
        if self_loop:
            node_in = node_out = torch.arange(graph.num_node, device=self.device)
            loop = torch.stack([node_in, node_out], dim=-1)
            edge_list = torch.cat([edge_list, loop])
            edge_weight = torch.cat([edge_weight, torch.ones(graph.num_node, device=self.device)])
        relation = torch.zeros(len(edge_list), 1, dtype=torch.long, device=self.device)
        edge_list = torch.cat([edge_list, relation], dim=-1)
        graph = type(graph)(edge_list, edge_weight=edge_weight, num_node=graph.num_node,
                            num_relation=1, meta_dict=graph.meta_dict, **graph.data_dict)
        return graph

    def forward(self, graph, h_index, t_index=None, r_index=None, all_loss=None, metric=None):

        if graph.num_relation:
            pass
            # don't transform to undirected for now
        else:
            graph = self.as_relational_graph(graph)
            # h_index = h_index.view(-1, 1)
            # t_index = t_index.view(-1, 1)
            # r_index = torch.zeros_like(h_index)

        output = self.bellmanford(graph, h_index)
        feature = output["node_feature"]#.transpose(0, 1)
        # index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        # feature = feature.gather(1, index)

        #score = self.mlp(feature).squeeze(-1)
        #feature = self.mlp(feature)
        #return score.view(shape)
        return feature  # (num_relations, dim)
    
# Custom NBFNetFull - we initialize each relation in its own graph (bs, num_nodes, dim) instead of 
# all relations in a batch getting initializes in a single graph, output shape is (bs, num_relations, dim)
class CustomNBFNetFull(CustomNBFNet):

    def __init__(self, learn_query=False, **kwargs):
        super().__init__(**kwargs)
        self.learn_query = learn_query
        if learn_query:
            self.learnable_q = nn.Embedding(1, self.dims[0])
    
    @utils.cached
    def bellmanford(self, graph, h_index, separate_grad=False):
        query = torch.ones(h_index.shape[0], self.dims[0], device=h_index.device, dtype=torch.float)
        if self.learn_query:
            query = self.learnable_q.weight.expand(h_index.shape[0], self.dims[0])
        index = h_index.unsqueeze(-1).expand_as(query)
        # important: DO label nodes uniquely per batch, initialize (batch_size) graphs
        boundary = torch.zeros(graph.num_node, *query.shape, device=self.device)
        boundary.scatter_add_(0, index.unsqueeze(0), query.unsqueeze(0))

        with graph.graph():
            graph.query = query
        with graph.node():
            graph.boundary = boundary

        hiddens = []
        layer_input = boundary

        for layer in self.layers:
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        return {
            "node_feature": hiddens[-1].transpose(1, 0),  # shape: (bs, num_rel, dim) 
        }


@R.register("models.RelBNFNet")
class RelNBFNet(RelationModel):

    def __init__(self, input_dim, hidden, 
                 num_layers=6, 
                 **kwargs):
        super(RelNBFNet, self).__init__(multirelational=True, **kwargs)

        self.input_dim = input_dim
        self.ablation_etypes = kwargs.get('ablation_etypes', False)

        self.model = CustomNBFNetFull(
            input_dim=input_dim,
            hidden_dims=[hidden] * num_layers,
            num_relation=4 if not self.ablation_etypes else None,
            aggregate_func="sum",
            layer_norm=True,
            short_cut=True,
            learn_query=kwargs.get('learn_query', False)
        )
        self.hidden_dim = hidden
        
        if self.hidden_dim != self.input_dim:
            self.input_transform_linear = nn.Linear(self.input_dim, self.hidden_dim)

    @_reset_input
    def forward(self, graph, input, r_idx, all_loss=None, metric=None):
        # TODO: consider adding RWSE features to all-ones
        # x = input.unsqueeze(0)

        x = self.model(graph, h_index=r_idx)

        return {
            "graph_feature": None,
            "node_feature": x  # [num_rel, dim] or [bs, num_rel, dim]
        }
    
from collections import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch import autograd

from torch_scatter import scatter_add, scatter_min

from torchdrug import core, layers, utils
from torchdrug.layers import functional
from torchdrug.core import Registry as R

from . import layer, functional

    
@R.register("models.TransferNBFNet") # NOTE
class TransferNBFNet(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dims, num_relation=None, symmetric=False,
                 message_func="distmult", aggregate_func="pna", short_cut=False, layer_norm=False, activation="relu",
                 concat_hidden=False, num_mlp_layer=2, project=True, remove_one_hop=False,
                 num_beam=10, path_topk=10, mod=False):
        super(TransferNBFNet, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        if num_relation is None:
            double_relation = 1
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
        layer_type = layer.GeneralizedRelationalConvNBF if not mod else layer.GeneralizedRelationalConvNBFMod

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layer_type(self.dims[i], self.dims[i + 1], double_relation,
                                                               self.dims[0], message_func, aggregate_func, layer_norm,
                                                               activation, project))

        feature_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1) + input_dim
        
        self.query = None        
        
        self.mlp = layers.MLP(feature_dim, [feature_dim] * (num_mlp_layer - 1) + [1])

        self.dist_embed = nn.Embedding(10, input_dim)

    def remove_easy_edges(self, graph, h_index, t_index, r_index=None):
        if self.remove_one_hop:
            h_index_ext = torch.cat([h_index, t_index], dim=-1)
            t_index_ext = torch.cat([t_index, h_index], dim=-1)
            if r_index is not None:
                any = -torch.ones_like(h_index_ext)
                pattern = torch.stack([h_index_ext, t_index_ext, any], dim=-1)
            else:
                pattern = torch.stack([h_index_ext, t_index_ext], dim=-1)
        else:
            if r_index is not None:
                pattern = torch.stack([h_index, t_index, r_index], dim=-1)
            else:
                pattern = torch.stack([h_index, t_index], dim=-1)
        pattern = pattern.flatten(0, -2)
        edge_index = graph.match(pattern)[0]
        edge_mask = ~layers.functional.as_mask(edge_index, graph.num_edge)
        return graph.edge_mask(edge_mask)

    def negative_sample_to_tail(self, h_index, t_index, r_index, num_relations):
        # convert p(h | t, r) to p(t' | h', r')
        # h' = t, r' = r^{-1}, t' = h
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + num_relations)
        return new_h_index, new_t_index, new_r_index

    def as_relational_graph(self, graph, self_loop=True):
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

    @utils.cached
    def bellmanford(self, graph, h_index, r_index, separate_grad=False):
        # query can be of shape (num_rel, dim) or (bs, num_rel, dim)
        bs = h_index.shape[0]
        query = self.query[r_index] if len(self.query.shape) == 2 else self.query[torch.arange(bs, device=graph.device), r_index]
        
        index = h_index.unsqueeze(-1).expand_as(query)
        boundary = torch.zeros(graph.num_node, *query.shape, device=self.device)
        boundary.scatter_add_(0, index.unsqueeze(0), query.unsqueeze(0))

        with graph.graph():
            graph.query = query    # queries from relations of shape (batch_size, dim)
        with graph.node():
            graph.boundary = boundary

        hiddens = []
        step_graphs = []
        layer_input = boundary

        for layer in self.layers:
            if separate_grad:
                step_graph = graph.clone().requires_grad_()
            else:
                step_graph = graph
            hidden = layer(step_graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            step_graphs.append(step_graph)
            layer_input = hidden
        
        #ipdb.set_trace()

        node_query = query.expand(graph.num_node, -1, -1)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "step_graphs": step_graphs,
        }

    def forward(self, graph, rel_query_list, h_index, t_index, r_index=None, all_loss=None, metric=None):
        if all_loss is not None:
            graph = self.remove_easy_edges(graph, h_index, t_index, r_index)
            
        self.query = rel_query_list[0] # NOTE
        if len(rel_query_list) > 1:
            assert len(rel_query_list) == len(self.layers) + 1
            for i in range(len(self.layers)):
                self.layers[i].relation = rel_query_list[i + 1]
        else:
            for i in range(len(self.layers)):
                self.layers[i].relation = rel_query_list[0]
        if metric is not None:
            metric["query_norm"] = self.query.detach().norm()
            metric["query_mean"] = self.query.detach().mean()
            metric["query_std"] = self.query.detach().std()

        shape = h_index.shape
        if graph.num_relation:
            # in the multi-graph case where graphs are of different num_relations
            num_relations = graph.num_relation
            graph = graph.undirected(add_inverse=True)
            h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index, num_relations)
        else:
            graph = self.as_relational_graph(graph)
            h_index = h_index.view(-1, 1)
            t_index = t_index.view(-1, 1)
            r_index = torch.zeros_like(h_index)

        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()
        output = self.bellmanford(graph, h_index[:, 0], r_index[:, 0])
        feature = output["node_feature"].transpose(0, 1)
        if metric is not None:
            metric["output_norm"] = feature.detach().norm()
            metric["output_mean"] = feature.detach().mean()
            metric["output_std"] = feature.detach().std()
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(1, index)

        if self.symmetric:
            assert (t_index[:, [0]] == t_index).all()
            output = self.bellmanford(graph, t_index[:, 0], r_index[:, 0], rel_query_list[1:])
            inv_feature = output["node_feature"].transpose(0, 1)
            index = h_index.unsqueeze(-1).expand(-1, -1, inv_feature.shape[-1])
            inv_feature = inv_feature.gather(1, index)
            feature = (feature + inv_feature) / 2

        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)



@R.register("models.NBFNet") # NOTE
class NeuralBellmanFordNetwork(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dims, num_relation=None, symmetric=False,
                 message_func="distmult", aggregate_func="pna", short_cut=False, layer_norm=False, activation="relu",
                 concat_hidden=False, num_mlp_layer=2, dependent=True, remove_one_hop=False,
                 num_beam=10, path_topk=10,
                 separate_remove_one_hop=False):
        super(NeuralBellmanFordNetwork, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        if num_relation is None:
            double_relation = 1
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
            self.layers.append(layer.GeneralizedRelationalConvNBF(self.dims[i], self.dims[i + 1], double_relation,
                                                               self.dims[0], message_func, aggregate_func, layer_norm,
                                                               activation, dependent))

        feature_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1) + input_dim

        self.query = nn.Embedding(double_relation, input_dim)
        
        self.mlp = layers.MLP(feature_dim, [feature_dim] * (num_mlp_layer - 1) + [1])

        self.dist_embed = nn.Embedding(10, input_dim)
        self.separate_remove_one_hop = separate_remove_one_hop

    def remove_easy_edges(self, graph, h_index, t_index, r_index=None):
        # remove_one_hop=True
        h_index_ext = torch.cat([h_index, t_index], dim=-1)
        t_index_ext = torch.cat([t_index, h_index], dim=-1)
        if r_index is not None:
            any = -torch.ones_like(h_index_ext)
            pattern1 = torch.stack([h_index_ext, t_index_ext, any], dim=-1)
        else:
            pattern1 = torch.stack([h_index_ext, t_index_ext], dim=-1)
        # remove_one_hop=False
        if r_index is not None:
            pattern2 = torch.stack([h_index, t_index, r_index], dim=-1)
        else:
            pattern2 = torch.stack([h_index, t_index], dim=-1)
        #ipdb.set_trace()
        
        if self.separate_remove_one_hop:
            # if separate_remove_one_hop is True: remove_one_hop only applies to 
            # the first subset of the merged dataset
            assert hasattr(self, "merge_info")
            gap = self.merge_info[0]["num_node"]
            mask = h_index[:, 0] < gap
            assert torch.all(mask == (t_index[:, 0] < gap))
            if self.remove_one_hop:
                # pattern 1 (< gap) + pattern 2
                pattern = torch.cat([pattern1[mask].flatten(0, -2), pattern2[~mask].flatten(0, -2)])
            else:
                # pattern 2 + pattern 1 (> gap)
                pattern = torch.cat([pattern2[mask].flatten(0, -2), pattern1[~mask].flatten(0, -2)])
            
        else:
            pattern = pattern1 if self.remove_one_hop else pattern2
            pattern = pattern.flatten(0, -2)

        edge_index = graph.match(pattern)[0]
        edge_mask = ~layers.functional.as_mask(edge_index, graph.num_edge)
        return graph.edge_mask(edge_mask)        

    def negative_sample_to_tail(self, h_index, t_index, r_index):
        # convert p(h | t, r) to p(t' | h', r')
        # h' = t, r' = r^{-1}, t' = h
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation)
        return new_h_index, new_t_index, new_r_index

    def as_relational_graph(self, graph, self_loop=True):
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
    
    def _get_shortest_distance(self, graph, h_index, num_iters=100):

        dist = torch.ones(graph.num_node, len(h_index), dtype=torch.int32, device=self.device) * graph.num_node
        for i, start in enumerate(h_index):
            dist[start, i] = 0
        for i in range(num_iters):

            node_in, node_out = graph.edge_list[:, :2].t()
            message = dist[node_in] + 1
            update, _ = scatter_min(message, node_out, dim=0, dim_size=graph.num_node)

            dist[node_out] = torch.min(dist[node_out], update[node_out])
        return dist

    @utils.cached
    def bellmanford(self, graph, h_index, r_index, separate_grad=False):
        query = self.query(r_index)
        index = h_index.unsqueeze(-1).expand_as(query)
        boundary = torch.zeros(graph.num_node, *query.shape, device=self.device)
        boundary.scatter_add_(0, index.unsqueeze(0), query.unsqueeze(0))

        '''dist = self._get_shortest_distance(graph, h_index)
        #import ipdb
        #ipdb.set_trace()
        max_dist = dist.unique()[-2].item()
        for i in range(1, max_dist):
            index = (dist == i)
            boundary[index] = self.dist_embed(torch.tensor([i], dtype=torch.int32, device=self.device))'''
        with graph.graph():
            graph.query = query
        with graph.node():
            graph.boundary = boundary

        hiddens = []
        step_graphs = []
        layer_input = boundary

        for layer in self.layers:
            if separate_grad:
                step_graph = graph.clone().requires_grad_()
            else:
                step_graph = graph
            hidden = layer(step_graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            step_graphs.append(step_graph)
            layer_input = hidden

        node_query = query.expand(graph.num_node, -1, -1)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "step_graphs": step_graphs,
        }

    def forward(self, graph, h_index, t_index, r_index=None, all_loss=None, metric=None):
        if all_loss is not None:
            graph = self.remove_easy_edges(graph, h_index, t_index, r_index)

        shape = h_index.shape
        if graph.num_relation:
            graph = graph.undirected(add_inverse=True)
            h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)
        else:
            graph = self.as_relational_graph(graph)
            h_index = h_index.view(-1, 1)
            t_index = t_index.view(-1, 1)
            r_index = torch.zeros_like(h_index)

        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()
        output = self.bellmanford(graph, h_index[:, 0], r_index[:, 0])
        feature = output["node_feature"].transpose(0, 1)
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(1, index)

        if self.symmetric:
            assert (t_index[:, [0]] == t_index).all()
            output = self.bellmanford(graph, t_index[:, 0], r_index[:, 0])
            inv_feature = output["node_feature"].transpose(0, 1)
            index = h_index.unsqueeze(-1).expand(-1, -1, inv_feature.shape[-1])
            inv_feature = inv_feature.gather(1, index)
            feature = (feature + inv_feature) / 2

        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)

    def visualize(self, graph, h_index, t_index, r_index):
        assert h_index.numel() == 1 and h_index.ndim == 1
        graph = graph.undirected(add_inverse=True)

        output = self.bellmanford(graph, h_index, r_index, separate_grad=True)
        feature = output["node_feature"]
        step_graphs = output["step_graphs"]

        index = t_index.unsqueeze(0).unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(0, index).squeeze(0)
        score = self.mlp(feature).squeeze(-1)

        edge_weights = [graph.edge_weight for graph in step_graphs]
        edge_grads = autograd.grad(score, edge_weights)
        for graph, edge_grad in zip(step_graphs, edge_grads):
            with graph.edge():
                graph.edge_grad = edge_grad
        distances, back_edges = self.beam_search_distance(step_graphs, h_index, t_index, self.num_beam)
        paths, weights = self.topk_average_length(distances, back_edges, t_index, self.path_topk)

        return paths, weights

    @torch.no_grad()
    def beam_search_distance(self, graphs, h_index, t_index, num_beam=10):
        num_node = graphs[0].num_node
        input = torch.full((num_node, num_beam), float("-inf"), device=self.device)
        input[h_index, 0] = 0

        distances = []
        back_edges = []
        for graph in graphs:
            graph = graph.edge_mask(graph.edge_list[:, 0] != t_index)
            node_in, node_out = graph.edge_list.t()[:2]

            message = input[node_in] + graph.edge_grad.unsqueeze(-1)
            msg_source = graph.edge_list.unsqueeze(1).expand(-1, num_beam, -1)

            is_duplicate = torch.isclose(message.unsqueeze(-1), message.unsqueeze(-2)) & \
                           (msg_source.unsqueeze(-2) == msg_source.unsqueeze(-3)).all(dim=-1)
            is_duplicate = is_duplicate.float() - \
                           torch.arange(num_beam, dtype=torch.float, device=self.device) / (num_beam + 1)
            # pick the first occurrence as the previous state
            prev_rank = is_duplicate.argmax(dim=-1, keepdim=True)
            msg_source = torch.cat([msg_source, prev_rank], dim=-1)

            node_out, order = node_out.sort()
            node_out_set = torch.unique(node_out)
            # sort message w.r.t. node_out
            message = message[order].flatten()
            msg_source = msg_source[order].flatten(0, -2)
            size = scatter_add(torch.ones_like(node_out), node_out, dim_size=num_node)
            msg2out = torch.repeat_interleave(size[node_out_set] * num_beam)
            # deduplicate
            is_duplicate = (msg_source[1:] == msg_source[:-1]).all(dim=-1)
            is_duplicate = torch.cat([torch.zeros(1, dtype=torch.bool, device=self.device), is_duplicate])
            message = message[~is_duplicate]
            msg_source = msg_source[~is_duplicate]
            msg2out = msg2out[~is_duplicate]
            size = scatter_add(torch.ones_like(msg2out), msg2out, dim_size=len(node_out_set))

            if not torch.isinf(message).all():
                distance, rel_index = functional.variadic_topk(message, size, k=num_beam)
                abs_index = rel_index + (size.cumsum(0) - size).unsqueeze(-1)
                back_edge = msg_source[abs_index]
                distance = distance.view(len(node_out_set), num_beam)
                back_edge = back_edge.view(len(node_out_set), num_beam, 4)
                distance = scatter_add(distance, node_out_set, dim=0, dim_size=num_node)
                back_edge = scatter_add(back_edge, node_out_set, dim=0, dim_size=num_node)
            else:
                distance = torch.full((num_node, num_beam), float("-inf"), device=self.device)
                back_edge = torch.zeros(num_node, num_beam, 4, dtype=torch.long, device=self.device)

            distances.append(distance)
            back_edges.append(back_edge)
            input = distance

        return distances, back_edges

    def topk_average_length(self, distances, back_edges, t_index, k=10):
        paths = []
        average_lengths = []

        for i in range(len(distances)):
            # TODO: are these two lines necessary
            distance, order = distances[i][t_index].flatten(0, -1).sort(descending=True)
            back_edge = back_edges[i][t_index].flatten(0, -2)[order]
            for d, (h, t, r, prev_rank) in zip(distance[:k].tolist(), back_edge[:k].tolist()):
                if d == float("-inf"):
                    break
                path = [(h, t, r)]
                for j in range(i - 1, -1, -1):
                    h, t, r, prev_rank = back_edges[j][h, prev_rank].tolist()
                    path.append((h, t, r))
                paths.append(path[::-1])
                average_lengths.append(d / len(path))

        if paths:
            average_lengths, paths = zip(*sorted(zip(average_lengths, paths), reverse=True)[:k])

        return paths, average_lengths
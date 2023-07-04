import numpy as np
import torch
from scipy.linalg import eigh
import warnings
import networkx as nx

from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.nn import GINConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import SplineConv

import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')
np.random.seed(0)


class GraphUnet(nn.Module):

    def __init__(self, ks, in_dim, out_dim, dim, act, drop_p):
        super(GraphUnet, self).__init__()
        self.ks = ks
        self.bottom_gcn = GCN(dim, dim, act, drop_p)
        self.down_gcns = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.down_gcns.append(GCN(dim, dim, act, drop_p))
            self.up_gcns.append(GCN(dim, dim, act, drop_p))
            self.pools.append(Pool(ks[i], dim, drop_p))
            self.unpools.append(Unpool(dim, dim, drop_p))

    def forward(self, g, h):
        adj_ms = []
        indices_list = []
        down_outs = []
        hs = []
        org_h = h
        for i in range(self.l_n):
            h = self.down_gcns[i](g, h)
            adj_ms.append(g)
            down_outs.append(h)
            g, h, idx = self.pools[i](g, h)
            indices_list.append(idx)
        h = self.bottom_gcn(g, h)
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            g, idx = adj_ms[up_idx], indices_list[up_idx]
            g, h = self.unpools[i](g, h, down_outs[up_idx], idx)
            h = self.up_gcns[i](g, h)
            h = h.add(down_outs[up_idx])
            hs.append(h)
        h = h.add(org_h)
        hs.append(h)
        return hs


def to_pyg_edgeindex(g):
    src_list = []
    dst_list = []
    attr_list = []
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            if g[i, j] > 0:
                src_list.append(i)
                dst_list.append(j)
                attr_list.append(g[i, j])
    final_list = [src_list, dst_list]
    return torch.tensor(final_list), torch.tensor(attr_list)


# GCN
# class GCN(nn.Module):
#
#     def __init__(self, in_dim, out_dim, act, p):
#         super(GCN, self).__init__()
#         self.proj = nn.Linear(in_dim, out_dim)
#         self.act = act
#         self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()
#
#     def forward(self, g, h):
#         # print('type h:'+str(type(h)))
#         # print('shape h:'+str(h.shape))
#         # print(h)
#         # print('type g:'+str(type(g)))
#         # print('shape g:'+str(g.shape))
#         # print(g)
#         h = self.drop(h)
#         h = torch.matmul(g, h)
#         h = self.proj(h)
#         h = self.act(h)
#         return h

# GAT
# class GCN(nn.Module):
#     def __init__(self, in_dim, out_dim, act, p):
#         super(GCN, self).__init__()
#         self.act = act
#         self.in_head = 8
#         self.out_head = 1
#         self.conv1 = GATConv(in_dim, out_dim, heads=self.in_head, dropout=p,concat=False)
#
#     def forward(self, g,h):
#         x = h
#         edge_index,edge_attr = to_pyg_edgeindex(g)
#         x = F.dropout(x, p=0.6, training=self.training)
#     x = self.conv1(x, edge_index)
#         #x = F.elu(x)
#         x = self.act(x)
#         return x

# GATv2
# class GCN(nn.Module):
#     def __init__(self, in_dim, out_dim, act, p):
#         super(GCN, self).__init__()
#         self.act = act
#         self.in_head = 8
#         self.out_head = 1
#         self.conv1 = GATv2Conv(in_dim, out_dim, heads=self.in_head, dropout=p,concat=False)
#
#     def forward(self, g,h):
#         x = h
#         edge_index,edge_attr = to_pyg_edgeindex(g)
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = self.conv1(x, edge_index)
#         #x = F.elu(x)
#         x = self.act(x)
#         return x

# GIN
class GCN(nn.Module):
    """
    Graph Isomorphism Network (GIN) layer
    """

    def __init__(self, in_dim, out_dim, act, p):
        super(GCN, self).__init__()
        self.act = act
        self.p = p
        gin_nn = nn.Sequential(Linear_pyg(in_dim, out_dim), nn.ReLU(), Linear_pyg(out_dim, out_dim))
        self.model = GINConv(gin_nn)

    def forward(self, g, h):
        x = h
        edge_index, _ = to_pyg_edgeindex(g)
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.model(x, edge_index)
        x = self.act(x)
        return x


# SAGEConv
# class GCN(nn.Module):
#     """
#     GraphSAGE Conv layer
#     """
#     def __init__(self,in_dim,out_dim,act,p):
#         super(GCN, self).__init__()
#         self.act = act
#         self.model = SAGEConv(in_dim,out_dim,bias=True)
#
#     def forward(self,g,h):
#         x = h
#         edge_index,edge_attr = to_pyg_edgeindex(g)
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = self.model(x, edge_index)
#         x = self.act(x)
#         return x


# TransformerConv
# class GCN(nn.Module):
#     """
#     GraphSAGE Conv layer
#     """
#     def __init__(self,in_dim,out_dim,act,p):
#         super(GCN, self).__init__()
#         self.act = act
#         self.model = TransformerConv(in_dim,out_dim,heads = 1,bias=True)
#
#     def forward(self,g,h):
#         x = h
#         edge_index,edge_attr = to_pyg_edgeindex(g)
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = self.model(x, edge_index)
#         x = self.act(x)
#         return x


class Pool(nn.Module):

    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        # self.matrix_power = 3
        # self.eigs_num = 4
        self.centralities_num = 6
        self.sigmoid = nn.Sigmoid()
        self.feature_proj = nn.Linear(in_dim, 1)
        self.structure_proj = nn.Linear(self.centralities_num, 1)
        # self.structure_proj = nn.Linear(self.eigs_num, 1)
        self.final_proj = nn.Linear(2, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        Z = self.drop(h)
        G = nx.from_numpy_array(g.detach().numpy())
        C = all_centralities(G).float()
        # L = torch.matrix_power(normalized_laplacian(g), self.matrix_power)
        # L = normalized_laplacian(g)
        # L_a = approximate_matrix(L, self.eigs_num)
        feature_weights = self.feature_proj(Z)
        structure_weights = self.structure_proj(C)
        # structure_weights = self.structure_proj(L_a)
        weights = torch.cat([feature_weights, structure_weights], dim=1)
        weights = self.final_proj(weights).squeeze()
        scores = self.sigmoid(weights)

        return top_k_graph(scores, g, h, self.k)


class Unpool(nn.Module):

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, g, h, pre_h, idx):
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h
        a = list(idx)
        b = list(range(g.shape[0]))
        new_list = [index for index in a if index not in b]
        idx_prime = torch.tensor(new_list)

        for i in idx_prime:
            tmp1 = np.zeros(h.shape[1])
            tmp2 = 0
            for j in range(g.shape[0]):
                if g[i, j] > 0:
                    tmp1 += h[j] * g[i, j]
                    tmp2 += g[i, j]
            new_h[i] = tmp1 / tmp2
        return g, new_h


def centrality_based(centrality_metric, graph, num_top):
    # these ones return a Dictionary of nodes with centrality as the value.
    if centrality_metric == 'closeness':
        centrality = nx.closeness_centrality(graph)
    elif centrality_metric == 'degree':
        centrality = nx.degree_centrality(graph)
    elif centrality_metric == 'eigenvector':
        centrality = nx.eigenvector_centrality(graph)
    elif centrality_metric == 'betweenness':
        centrality = nx.betweenness_centrality(graph)
    elif centrality_metric == 'load':
        centrality = nx.load_centrality(graph)
    elif centrality_metric == 'subgraph':
        centrality = nx.subgraph_centrality(graph)
    elif centrality_metric == 'harmonic':
        centrality = nx.harmonic_centrality(graph)
    return torch.tensor(np.array(list(centrality.values())))


def all_centralities(graph):
    num_nodes = graph.number_of_nodes()
    centrality = np.zeros((num_nodes, 6))
    centrality[:, 0] = np.array(list(nx.closeness_centrality(graph).values()))
    centrality[:, 1] = np.array(list(nx.degree_centrality(graph).values()))
    centrality[:, 2] = np.array(list(nx.betweenness_centrality(graph)))
    centrality[:, 3] = np.array(list(nx.load_centrality(graph)))
    centrality[:, 4] = np.array(list(nx.subgraph_centrality(graph)))
    centrality[:, 5] = np.array(list(nx.harmonic_centrality(graph)))
    return torch.tensor(centrality)


def approximate_matrix(a, k):
    b = np.zeros((a.shape[0], k))
    _, v = eigh(a, subset_by_index=[0, min(k, a.shape[0]) - 1])
    for i in range(v.shape[1]):
        b[:, i] = v[:, i]
    return torch.tensor(np.single(b))


def normalized_laplacian(adjacency_matrix: torch.Tensor) -> torch.Tensor:
    """ Computes the symmetric normalized Laplacian matrix """
    num_nodes = adjacency_matrix.shape[0]
    d = torch.sum(adjacency_matrix, dim=1)
    for i in range(len(d)):
        if d[i] != 0:
            d[i] = d[i] ** (-0.5)
    Dinv = torch.diag(d)
    Ln = torch.eye(len(d)) - torch.matmul(torch.matmul(Dinv, adjacency_matrix), Dinv)
    # make sure the Laplacian is symmetric
    Ln = 0.5 * (Ln + Ln.T)
    return Ln


def top_k_graph(scores, g, h, k):
    num_nodes = g.shape[0]
    values, idx = torch.topk(scores, max(2, int(k * num_nodes)))
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)
    un_g = g.bool().float()
    un_g_2 = torch.matmul(un_g, un_g).float()
    un_g_3 = torch.matmul(un_g_2, un_g).float()
    un_g = (un_g + un_g_2 + un_g_3).bool().float()
    un_g = un_g[idx, :]
    un_g = un_g[:, idx]
    g = norm_g(un_g)
    return g, new_h, idx


def add_edges(adj_mat, threshold):
    adj_mat = adj_mat.detach().numpy()
    G = nx.from_numpy_array(adj_mat)
    complete_graph = nx.complete_graph(adj_mat.shape[0])
    preds = nx.jaccard_coefficient(G, complete_graph.edges())
    for u, v, p in preds:
        if p > threshold:
            G.add_edge(u, v)
    new_adj_mat = nx.to_numpy_array(G)
    return torch.tensor(new_adj_mat)


def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g


class Initializer(object):

    @classmethod
    def _glorot_uniform(cls, w):
        if len(w.size()) == 2:
            fan_in, fan_out = w.size()
        elif len(w.size()) == 3:
            fan_in = w.size()[1] * w.size()[2]
            fan_out = w.size()[0] * w.size()[2]
        else:
            fan_in = np.prod(w.size())
            fan_out = np.prod(w.size())
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        w.uniform_(-limit, limit)

    @classmethod
    def _param_init(cls, m):
        if isinstance(m, nn.parameter.Parameter):
            cls._glorot_uniform(m.data)
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
            cls._glorot_uniform(m.weight.data)

    @classmethod
    def weights_init(cls, m):
        for p in m.modules():
            if isinstance(p, nn.ParameterList):
                for pp in p:
                    cls._param_init(pp)
            else:
                cls._param_init(p)

        for name, p in m.named_parameters():
            if '.' not in name:
                cls._param_init(p)

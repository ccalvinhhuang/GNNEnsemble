import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import random
class MLP(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_feats, hidden_feats)
        self.linear2 = nn.Linear(hidden_feats, out_feats)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class GraphSAGELayer(nn.Module):
    def __init__(self, in_feats, out_feats, aggregator_type='mean', feat_drop=0.0, bias=True):
        super(GraphSAGELayer, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.feat_drop = nn.Dropout(feat_drop)
        #self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        #self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        self.mlp_list = nn.ModuleList([MLP(self._in_src_feats, out_feats, out_feats, bias=False) for _ in range(20)])
        self.reset_parameters()


    def reset_parameters(self):

        gain = nn.init.calculate_gain("relu")
        if self._aggre_type == "pool":
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == "lstm":
            self.lstm.reset_parameters()
        gain = nn.init.calculate_gain("relu")
        if self._aggre_type != "gcn":
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        # nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
        for mlp in self.mlp_list:
            nn.init.xavier_uniform_(mlp.linear1.weight, gain=gain)
            nn.init.xavier_uniform_(mlp.linear2.weight, gain=gain)


    def forward(self, graph, feat):
        with graph.local_scope():
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                msg_fn = fn.copy_u("h", "m")
                h_self = feat_dst

                if graph.num_edges() == 0:
                    graph.dstdata["neigh"] = torch.zeros(
                        feat_dst.shape[0], self._in_src_feats
                    ).to(feat_dst)

                lin_before_mp = self._in_src_feats > self._out_feats

                if self._aggre_type == "mean":
                    graph.srcdata["h"] = (
                        self.fc_neigh(feat_src) if lin_before_mp else feat_src
                    )
                    graph.update_all(msg_fn, fn.mean("m", "neigh"))
                    h_neigh = graph.dstdata["neigh"]
                    if not lin_before_mp:
                        chosen_mlp = random.choice(self.mlp_list)
                        h_neigh = chosen_mlp(h_neigh)
                h_self = self.fc_self(h_self)
                rst = h_self + h_neigh
                return rst


def expand_as_pair(input_, g=None):
    if isinstance(input_, tuple):
        return input_
    elif g is not None and g.is_block:
        if isinstance(input_, Mapping):
            input_dst = {
                k: F.narrow_row(v, 0, g.number_of_dst_nodes(k))
                for k, v in input_.items()
            }
        else:
            input_dst = F.narrow_row(input_, 0, g.number_of_dst_nodes())
        return input_, input_dst
    else:
        return input_, input_

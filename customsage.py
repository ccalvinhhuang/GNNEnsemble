import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

class GraphSAGELayer(nn.Module):
    def __init__(self, in_feats, out_feats, aggregator_type='mean', feat_drop=0.0, bias=True):
        super(GraphSAGELayer, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)


        self.linear1 = nn.Linear(in_feats, out_feats, bias=bias)
        self.linear2 = nn.Linear(in_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear1.weight.size(1))

        self.linear1.weight.data.uniform_(-stdv, stdv)
        self.linear2.weight.data.uniform_(-stdv, stdv)
        if self.linear1.bias is not None:
            self.linear1.bias.data.uniform_(-stdv, stdv)
            self.linear2.bias.data.uniform_(-stdv, stdv)

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
            rst = self.fc_self(h_self) + h_neigh
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
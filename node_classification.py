import argparse
import copy
import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, RedditDataset, FlickrDataset
import linearsage as te
import customsage as tr
import testsage as ts
from dgl import AddSelfLoop
import matplotlib.pyplot as plt
class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(tr.GraphSAGELayer(in_size, hid_size, "mean"))
        self.layers.append(tr.GraphSAGELayer(hid_size, hid_size, "mean", True, True))
        self.layers.append(tr.GraphSAGELayer(hid_size, out_size, "mean"))

        #self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        #self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        #self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))

        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                dtype=feat.dtype,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0]: output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y
def evaluate(model, graph, dataloader, num_classes):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )


def layerwise_infer(device, graph, nid, model, num_classes, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(
            graph, device, batch_size
        )  # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata["label"][nid].to(pred.device)
        return MF.accuracy(
            pred, label, task="multiclass", num_classes=num_classes
        )

def train(args, device, g, dataset, model, num_classes):
    # create sampler & dataloader
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)
    sampler = NeighborSampler(
        [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    sampler_validation = NeighborSampler(
        [1000000, 1000000, 1000000],  # fanout for [layer-0, layer-1, layer-2]
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )

    use_uva = args.mode == "mixed"
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler_validation,
        device=device,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    best_model = None
    best_acc = 0

    epoch_accuracies = []
    for epoch in range(75):
        if epoch == 20:
            for i in range(len(model.layers)):
                for j in range(len(model.layers[i].mlp_list)):
                    #add noise to current MLP (model.layers[i].mlp_list[j])
                    for name, param in model.layers[i].mlp_list[j].named_parameters():
                        if 'weight' in name:
                            cur_norm = torch.norm(param.data)
                            noise = torch.randn_like(param.data).to("cuda:0") * 0.01 * cur_norm
                            param.data.add_(noise)

                    copies = 2
                    for k in range(copies):
                        mlp_copy = copy.deepcopy(model.layers[i].mlp_list[j])
                        #add noise to the mlp_copy
                        for name, param in mlp_copy.named_parameters():
                            if 'weight' in name:
                                cur_norm = torch.norm(param.data)
                                noise = torch.randn_like(param.data).to("cuda:0") * 0.01 * cur_norm
                                param.data.add_(noise)

                        model.layers[i].mlp_list.append(mlp_copy)

        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(
                train_dataloader
        ):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        acc = layerwise_infer(device, g, dataset.val_idx, model, num_classes, batch_size=4096)
        # acc = evaluate(model, g, val_dataloader, num_classes)
        if acc > best_acc:
            best_acc = acc
            best_model = copy.deepcopy(model)

        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Best Accuracy {:.4f}".format(
                epoch, total_loss / (it + 1), acc.item(), best_acc.item()
            )
        )
        epoch_accuracies.append(best_acc.item())
    layerwise_infer(device, g, dataset.test_idx, best_model, num_classes, batch_size=4096)
    return epoch_accuracies
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["cpu", "mixed", "puregpu"],
        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
    )
    parser.add_argument(
        "--dt",
        type=str,
        default="float",
        help="data type(float, bfloat16)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=float,
        default=256
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")

    # load and preprocess dataset
    print("Loading data")
    #dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))
    transform = (
        AddSelfLoop()
    )
    dataset = AsNodePredDataset(CoraGraphDataset(transform=transform))
    #dataset = AsNodePredDataset(CiteseerGraphDataset(transform=transform))
    #dataset = AsNodePredDataset(FlickrDataset(transform=transform))
    #dataset = AsNodePredDataset(RedditDataset(raw_dir = "/data/calvin/dgl", transform=transform))
    g = dataset[0]
    g = g.to("cuda" if args.mode == "puregpu" else "cpu")
    num_classes = dataset.num_classes
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")
    print(g)
    # create GraphSAGE model
    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    model = SAGE(in_size, args.hidden_dim, out_size).to(device)

    # convert model and graph to bfloat16 if needed
    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        model = model.to(dtype=torch.bfloat16)

    # model training
    print("Training...")
    epoch_accuracies = train(args, device, g, dataset, model, num_classes)
    # test the model
    print("Testing...")
    acc = layerwise_infer(
        device, g, dataset.test_idx, model, num_classes, batch_size=4096
    )
    print("Test Accuracy {:.4f}".format(acc.item()))
    plt.plot(epoch_accuracies)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs')
    # plt.savefig('accuracy_plot.png')






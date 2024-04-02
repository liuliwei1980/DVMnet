import numpy as np
import torch.nn as nn
from torch_geometric.nn import GATConv, SGConv, FiLMConv, GATv2Conv,GCNConv
import itertools
import torch
from GAT import GAT


class DVMNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.conv2l = nn.Linear(64, 64)
        self.dilin1 = nn.Linear(64, 64)
        self.dilin2 = nn.Linear(64, 9)
        self.sublin1 = nn.Linear(64, 64)
        self.sublin2 = nn.Linear(64, 6)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.GraphA = GAT(12, 30, 30, 0.2, 0.2, 4)
        self.GraphB = GAT(8, 30, 30, 0.2, 0.2, 4)
        self.BN30A = nn.BatchNorm1d(30)
        self.BN30B = nn.BatchNorm1d(30)
        self.relu = nn.ReLU()
        self.x_gat_lnc1 = nn.Linear(30, 1)
        self.x_gat_lnc2 = nn.Linear(64, 100)
        self.x_gat_mi1 = nn.Linear(30, 1)
        self.x_gat_mi2 = nn.Linear(16, 100)

    def mainNet(self, x, x_lnc, x_mi, edge_index, xname, edge_label_index, di,sub):
        node_mi = self.mer_x(2).cuda(0)
        node_lnc = self.mer_x(3).cuda(0)
        # Inner graph neural network LncRNA
        x_gat_lnc = torch.zeros(284, 64, 30)
        for i in range(x_lnc.size()[0]):
            a = x_lnc[i]
            a[a <= 0.03] = 0
            e = node_lnc
            x_gat_1 = self.GraphA(e, a)
            x_gat_1 = self.BN30A(x_gat_1)
            x_gat_1 = self.relu(x_gat_1)
            x_gat_lnc[i] = x_gat_1
        x_gat_lnc = self.x_gat_lnc1(x_gat_lnc.cuda(0))
        x_gat_lnc = x_gat_lnc.squeeze()
        x_gat_lnc = self.x_gat_lnc2(x_gat_lnc.cuda(0))
        x_gat_lnc = x_gat_lnc.squeeze()
        # Inner graph neural network miRNA
        x_gat_mi = torch.zeros(520, 16, 30)
        for i in range(x_mi.size()[0]):
            a = x_mi[i]
            a[a <= 0.03] = 0
            e = node_mi
            x_gat_1 = self.GraphB(e, a)
            x_gat_1 = self.BN30B(x_gat_1)
            x_gat_1 = self.relu(x_gat_1)
            x_gat_mi[i] = x_gat_1
        #outer graph neural network
        x_gat_mi = self.x_gat_mi1(x_gat_mi.cuda(0))
        x_gat_mi = x_gat_mi.squeeze()
        x_gat_mi = self.x_gat_mi2(x_gat_mi)
        x_cat = torch.cat((x_gat_lnc, x_gat_mi), dim=0)
        x_cat = self.relu(x_cat)
        x = x + x_cat * 0.2
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        out = (x[edge_label_index[0]] * x[edge_label_index[1]])
        out = out.sum(dim=-1)
        #task2
        diname = di[:, 0]
        di_out = di[:, 1:]
        di_out = di_out.astype(np.float64)
        di_out = torch.from_numpy(di_out)
        set1 = set(xname)
        indices = [xname.index(item) for item in diname if item in set1]
        di_x = x[indices]
        di_x = self.dilin1(di_x)
        di_x = self.relu(di_x)
        di_x = self.dilin2(di_x)
        di_x = self.sig(di_x)
        #task3
        subname = sub[:, 0]
        sub_out = sub[:, 1:]
        sub_out = sub_out.astype(np.float64)
        sub_out = torch.from_numpy(sub_out)
        set1 = set(xname)
        subindices = [xname.index(item) for item in subname if item in set1]
        sub_x = x[subindices]
        sub_x = self.sublin1(sub_x)
        sub_x = self.relu(sub_x)
        sub_x = self.sublin2(sub_x)
        sub_x = self.sig(sub_x)
        return out, di_x, di_out,sub_x,sub_out

    def mer_x(self, k):
        nucleotides = ['A', 'C', 'G', 'T']
        nucleotide_to_onehot = {
            'A': [1, 0, 0, 0],
            'C': [0, 1, 0, 0],
            'G': [0, 0, 1, 0],
            'T': [0, 0, 0, 1]
        }
        kmers = [''.join(combo) for combo in itertools.product(nucleotides, repeat=k)]
        onehot_encoded_kmers = torch.tensor([
            [nucleotide_to_onehot[kmer[i]] for i in range(len(kmer))] for kmer in kmers
        ], dtype=torch.float32)
        n_kmers = len(kmers)
        flattened_tensor_kmers = onehot_encoded_kmers.view(n_kmers, -1)
        return flattened_tensor_kmers





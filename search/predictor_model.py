# https://github.com/ultmaster/neuralpredictor.pytorch/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0")

def normalize_adj_batch(adj):
    # Row-normalize matrix
    last_dim = adj.size(-1)
    rowsum = adj.sum(2, keepdim=True).repeat(1, 1, last_dim)

    return torch.div(adj, rowsum)


def graph_pooling_batch(inputs, num_vertices):
    out = inputs.sum(1)

    return torch.div(out, num_vertices)


def normalize_adj(adj):
    # Row-normalize matrix
    last_dim = adj.size(-1)
    rowsum = adj.sum(1, keepdim=True).repeat(1, last_dim)

    return torch.div(adj, rowsum)


def graph_pooling(inputs, num_vertices):
    out = inputs.sum(0)

    return torch.div(out, num_vertices)


class DirectedGraphConvolutionLayerresetting(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight1 = nn.Parameter(torch.zeros((in_features, out_features)))
        self.weight2 = nn.Parameter(torch.zeros((in_features, out_features)))
        self.dropout = nn.Dropout(0.1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight1.data)
        nn.init.xavier_uniform_(self.weight2.data)

    def forward(self, inputs, adj):
        norm_adj = normalize_adj(adj)

        output1 = F.relu(torch.matmul(norm_adj, torch.matmul(inputs, self.weight1)))
        inv_norm_adj = normalize_adj(adj.transpose(0, 1))
        output2 = F.relu(torch.matmul(inv_norm_adj, torch.matmul(inputs, self.weight2)))
        
        out = (output1 + output2) / 2
        out = self.dropout(out)
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class NeuralPredictor(nn.Module):
    def __init__(self, initial_hidden=8, gcn_hidden=144, gcn_layers=3, linear_hidden=256):
        super().__init__()
        self.gcn = [DirectedGraphConvolutionLayerresetting(initial_hidden if i == 0 else gcn_hidden, gcn_hidden)
                    for i in range(gcn_layers)]
        self.gcn = nn.ModuleList(self.gcn)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

        self.embedding = nn.Embedding(num_embeddings= 3, embedding_dim=36)

        self.lstm = nn.LSTMCell(gcn_hidden, gcn_hidden)

        self.fc1 = nn.Linear(gcn_hidden+36, linear_hidden)
        self.fc2 = nn.Linear(linear_hidden, linear_hidden)
        self.fc3 = nn.Linear(linear_hidden, 1)


    def forward(self, numv, adj, out, hidden, cur_stage):
        numv = torch.tensor(numv, dtype=torch.int, device=device)
        gs = adj.size(0)
        adj_with_diag = normalize_adj(adj + torch.eye(gs, device=adj.device))  # assuming diagonal is not 1

        for layer in self.gcn:
            out = layer(out, adj_with_diag)

        out = graph_pooling(out, numv)
        out = out.unsqueeze(0)
        hx, cx = self.lstm(out, hidden)
        embedded = self.embedding(torch.tensor(cur_stage, device=device))
        embedded = embedded.unsqueeze(0)

        out = torch.cat((hx, embedded), dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out).view(-1)

        return out, (hx, cx)


class DirectedGraphConvolutionLayerresettingFID(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight1 = nn.Parameter(torch.zeros((in_features, out_features)))
        self.weight2 = nn.Parameter(torch.zeros((in_features, out_features)))
        self.dropout = nn.Dropout(0.1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight1.data)
        nn.init.xavier_uniform_(self.weight2.data)

    def forward(self, inputs, adj):
        norm_adj = normalize_adj(adj)
        output1 = F.relu(torch.matmul(norm_adj, torch.matmul(inputs, self.weight1)))
        inv_norm_adj = normalize_adj(adj.transpose(0, 1))
        output2 = F.relu(torch.matmul(inv_norm_adj, torch.matmul(inputs, self.weight2)))
        
        out = (output1 + output2) / 2
        out = self.dropout(out)
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class NeuralPredictorFID(nn.Module):
    def __init__(self, initial_hidden=8, gcn_hidden=144, gcn_layers=3, linear_hidden=256):
        super().__init__()
        self.gcn = [DirectedGraphConvolutionLayerresettingFID(initial_hidden if i == 0 else gcn_hidden, gcn_hidden)
                    for i in range(gcn_layers)]
        self.gcn = nn.ModuleList(self.gcn)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

        self.embedding = nn.Embedding(num_embeddings= 3, embedding_dim=36)

        self.lstm = nn.LSTMCell(gcn_hidden, gcn_hidden)

        self.fc1 = nn.Linear(gcn_hidden+36, linear_hidden)
        self.fc2 = nn.Linear(linear_hidden, linear_hidden)
        self.fc3 = nn.Linear(linear_hidden, 1)


    def forward(self, fid_score, numv, adj, out, hidden, cur_stage):
        numv = torch.tensor(numv, dtype=torch.int, device=device)
        gs = adj.size(0)
        adj_with_diag = normalize_adj(adj + torch.eye(gs, device=adj.device))  # assuming diagonal is not 1
        
        for layer in self.gcn:
            out = layer(out, adj_with_diag)

        out = graph_pooling(out, numv)
        out = out.unsqueeze(0)
        hx, cx = self.lstm(out, hidden)
        embedded = self.embedding(torch.tensor(cur_stage, device=device))
        embedded = embedded.unsqueeze(0)
        out = torch.cat((hx, embedded), dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out).view(-1)

        return out, (hx, cx)

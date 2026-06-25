import torch
import torch.nn as nn
import torch.nn.functional as F

class GAT(nn.Module):
    def __init__(self, d_feat=5, hidden_size=64, num_layers=2, dropout=0.0, num_graph_layer=2):
        super(GAT, self).__init__()
        
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        
        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.gat_layers = nn.ModuleList([
            GATLayer(hidden_size, hidden_size, dropout=dropout)
            for _ in range(num_graph_layer)
        ])
        
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x, relation_matrix=None):
        out, _ = self.rnn(x)
        hidden = out[:, -1, :]
        
        if relation_matrix is None:
            return self.fc(hidden)
        
        for gat_layer in self.gat_layers:
            hidden = gat_layer(hidden, relation_matrix)
        
        return self.fc(hidden)

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, Wh)
        
        return F.elu(h_prime)
    
    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]
        
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        
        all_combinations_matrix = torch.cat(
            [Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1
        )
        
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

def build_correlation_matrix(price_data, threshold=0.5):
    import numpy as np
    
    corr_matrix = np.corrcoef(price_data)
    
    relation_matrix = np.where(np.abs(corr_matrix) > threshold, np.abs(corr_matrix), 0.0)
    
    np.fill_diagonal(relation_matrix, 1.0)
    
    return relation_matrix

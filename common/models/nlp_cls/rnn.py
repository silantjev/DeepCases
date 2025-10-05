import torch
from torch import nn

from .common import make_mask

class Attention(nn.Module):
    """ Слой для внимания после RNN """
    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.multihead = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)

    def forward(self, x, query):
        x, lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        mask = make_mask(x.shape, lengths).to(x.device)
        assert query.dim() == x.dim() == 3
        assert query.shape[1] == 1
        assert query.shape[0] == x.shape[0]
        assert query.shape[2] == x.shape[2]
        output, _ = self.multihead(query, x, x, key_padding_mask=mask)
        return (output + query).squeeze(1) # residual connection


class RNNModel(nn.Module):
    """ RNN-классификатор с вниманием и без """
    def __init__(self,
                 embedding_dim, # разнерность входных векторов
                 output_dim,    # количество классов
                 hidden_dim,    # внутренняя размерность в RNN
                 n_layers,      # количество RNN-слоёв
                 n_heads=0,     # количество голов внимания
                 dropout=0.0,   # p_dropout для RNN и для внимания
                 dropout_lin=0.0, # p_droupout для линейного слоя
                 rnn_class='GRU', # тип RNN-слоёв: RNN, LSTM, GRU
                 bidirectional=False,
                 two_dense=False, # два линейных слоя в финале
            ):
        super().__init__()
        if isinstance(rnn_class, str):
            rnn_class = getattr(nn, rnn_class)
        self.name = rnn_class.__name__
        self.D = 1 + bidirectional
        rnn_dropout = dropout if n_layers > 1 else 0
        self.rnn = rnn_class(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bias=True,
            dropout=rnn_dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        if n_heads > 0: # с вниманием
            assert not bidirectional, "Bidirectional RNN with attention is not supported"
            self.attention = Attention(hidden_dim, n_heads=n_heads)
        self.dropout_after_rnn = nn.Dropout(dropout_lin)
        self.two_dense = two_dense
        if self.two_dense:
            dim0 = hidden_dim*self.D // 4
            self.fc0 = nn.Linear(hidden_dim*self.D, dim0)
            self.act0 = nn.ReLU()
            self.dropout_after_linear = nn.Dropout(dropout_lin)
            self.fc_final = nn.Linear(dim0, output_dim)
        else:
            self.fc_final = nn.Linear(hidden_dim*self.D, output_dim)
        
    def forward(self, x):
        x, h = self.rnn(x)
        if isinstance(h, tuple):
            h = h[0]
        if hasattr(self, 'attention'):
            x = self.attention(x, h[-1].unsqueeze(1))
        else:
            if self.D == 1:
                x = h[-1] # last layer
            else:
                assert self.D == 2
                # Cocatenate forward and backward states of the last layer
                x = torch.cat((h[-2], h[-1]), dim=1)

        x = self.dropout_after_rnn(x)
        if self.two_dense:
            x = self.fc0(x)
            x = self.act0(x)
            x = self.dropout_after_linear(x)
        x = self.fc_final(x)
        return x

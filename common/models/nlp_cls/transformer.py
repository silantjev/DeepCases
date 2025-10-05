# Классификатор на трансформере: TransformerClassifier
import math
import torch
from torch import nn

from .common import make_mask

class AddCLS(nn.Module):
    """Слой для добавления токена [CLS] в начало последовательности"""
    def __init__(self, emb_dim):
        super().__init__()
        # Создаем обучаемый параметр для токена [CLS]
        self.cls_token = nn.Parameter(torch.empty(emb_dim))
        # Инициализируем как в BERT (нормальное распределение с std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
    
    def forward(self, x, lengths):
        """
            Преобразовать тензор x формы [batch_size, seq_len, emb_dim]
            В тензор формы [batch_size, seq_len+1, emb_dim], добавив [CLS] в начале
            Также увеличивает элементы lengths на единицу
        """
        batch_size = x.size(0)
        
        # Тензор [CLS] для всего батча: [batch_size, 1, emb_dim]
        cls_tokens = self.cls_token.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)
        
        # Конкатенация с исходными данными по dim=1
        return torch.cat([cls_tokens, x], dim=1), lengths + 1


class PositionalEncoding(nn.Module):
    """Синусоидальное позиционное кодирование (необучаемое)"""
    def __init__(self, embedding_dim, max_len):
        super().__init__()
        # Вычисляем множитель для частот (10000^(-2*i/embedding_dim))
        factor = torch.exp(torch.arange(0, embedding_dim, 2).float() * 
                            (-math.log(10000.0) / embedding_dim))
        
        pe = torch.zeros(max_len, embedding_dim) # матрица позиционных кодов
        # Заполняем чётные индексы синусом, нечётные - косинусом
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        pe[:, 0::2] = torch.sin(position * factor)
        pe[:, 1::2] = torch.cos(position * factor)
        
        # Регистрируем как буфер (не обучаемый параметр)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch_size, seq_len, embedding_dim)
        seq_len = x.size(1)
        # Добавляем позиционные коды (берём только нужную длину последовательности)
        x += self.pe[:seq_len, :].unsqueeze(0)
        return x


class TransformerClassifier(nn.Module):
    """ Классификатор последовательностей,
        основанный на энкодере из 'Attention Is All You Need'
    """
    def __init__(self, num_classes, embedding_dim, n_heads, num_layers, dim_feedforward, max_seq_len,
            dropout=0.0, dropout_lin=0.0, add_cls=True, final_attention=True):
        super().__init__()

        if add_cls: # добавлять [CLS]
            self.cls_adder = AddCLS(embedding_dim)
            max_seq_len += 1

        self.pos_encoder = PositionalEncoding(embedding_dim, max_seq_len)
        
        # Подслои трансформера
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # В конце вместо ещё последнего подслоя трансформера эффективней добавить просто внимание
        if final_attention:
            assert add_cls, "final_attention can not be True if add_cls is False"
            self.multihead = nn.MultiheadAttention(
                    embed_dim=embedding_dim, num_heads=n_heads,
                    batch_first=True, dropout=dropout,
                )
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_lin),
            nn.Linear(embedding_dim // 2, num_classes)
        )
    
    def forward(self, x):
        x, lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        # Здесь x имеет форму [batch_size, seq_len, embedding_dim]
        if hasattr(self, 'cls_adder'):
            x, lengths = self.cls_adder(x, lengths)

        x = self.pos_encoder(x)

        mask = make_mask(x.shape, lengths).to(x.device)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        if hasattr(self, 'multihead'):
            query=x[:,0:1]
            x, _ = self.multihead(query, x, x, key_padding_mask=mask)
            x = (x + query).squeeze(1)
        elif hasattr(self, 'cls_adder'):
            x = x[:, 0, :]
        else:
            x = x.mean(dim=1)
        # Здесь x имеет форму [batch_size, embedding_dim]
        return self.classifier(x)


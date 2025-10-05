import torch
from torch import nn

def make_mask(shape, lengths):
    """ Общая функция для создания маски паддинга """
    batch_size, max_len, _ = shape
    return torch.arange(max_len).expand(batch_size, max_len) >= lengths.unsqueeze(1)


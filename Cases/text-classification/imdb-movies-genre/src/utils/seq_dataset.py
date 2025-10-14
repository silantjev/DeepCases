import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence

from .imdb_data_manager import IMDBDataManager

class SeqDataset(Dataset):
    """ Датасет для обучения и валидации
        Поскольку RNN-модели способны принимать PackedSequence,
        метод __getitem__ выдаёт данные в соответствующем виде
        На вход принимает 3 одномерных массива с целочисленными данными:
            X — склеенные в один массив последовательности кодов токенов
            lengths — информация о длине последовательностей, с помощью которой делится массив X
            y — закодированные метки классов 
        а также массив векторов vectors: токену с кодом i соответствует vectors[i]
        параметр compute_y_weights указывает, что надо посчитать веса классов, необходимых для loss-функции
    """
    def __init__(self, X, lengths, y, vectors, compute_y_weights=True):
        assert X.ndim == 1
        assert lengths.ndim == 1
        assert len(X) == lengths.sum()
        assert len(lengths) == len(y)
        if compute_y_weights:
            classes=np.unique(y)
            assert (classes == np.arange(len(classes))).all()
            class_weights = compute_class_weight('balanced', classes=classes, y=y)
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.emb = torch.nn.Embedding.from_pretrained(torch.from_numpy(vectors), freeze=True)
        self.embedding_dim = vectors.shape[1]
        self.y = torch.from_numpy(y)
        self.X = torch.from_numpy(X)
        self.max_length = np.max(lengths)
        lengths = torch.from_numpy(lengths)
        self.pointers = torch.cat([torch.tensor([0]), torch.cumsum(lengths, 0)])
        assert len(self.pointers) == len(self.y) + 1
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        start = self.pointers[idx]
        end = self.pointers[idx + 1]
        return self.emb(self.X[start:end]), self.y[idx]

def packed_collate_fn(batch):
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    sequences, y = zip(*batch)
    return pack_sequence(sequences, enforce_sorted=True), torch.stack(y)

def make_dataloader(path, vector_path=None, batch_size=None, num_workers=0, for_train=True, pin_memory=False, shuffle=None):
    if batch_size is None:
        batch_size = 32 if for_train else 128
    data_manager = IMDBDataManager()
    path = data_manager.make_abs(path)
    if vector_path is None:
        vector_path = path.parent / (path.stem + '_vectors.npy')
    else:
        vector_path = data_manager.make_abs(vector_path)
    assert vector_path.exists()
    X, lengths, y = data_manager.load_xy(path)
    vectors = np.load(vector_path)
    dataset = SeqDataset(X, lengths, y, vectors, compute_y_weights=for_train)
    if shuffle is None:
        shuffle = for_train
    dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=packed_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return dataloader

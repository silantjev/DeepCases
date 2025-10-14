from pathlib import Path
import numpy as np

#local imports
from split_data import read_conf
from utils.imdb_data_manager import IMDBDataManager
from utils.genre_encoder import GenreEncoder
from utils.tokenizator import Tokenizer
from utils.vocab import make_vocab, Vocab
from utils.cli_args import take_args

class DataPreprocessor:
    """ Предобработчик данных.
        Обучается во время первой предобработки train-данных
        Предполагает сохранение своего состояние в файлы
    """
    def __init__(self, txt_path, unk_threshold=0.1):
        if unk_threshold > 0:
            self.threshold = unk_threshold # minimal number of known tokens per text
        else:
            self.threshold = 2 # do no filter
        self.genre_encoder = GenreEncoder(txt_path=txt_path)
        self.tokenizer = Tokenizer()
        self.vocab = None
        self.vectors = None

    def save_state(self, name):
        assert self.fitted
        data_manager = IMDBDataManager(txt_path=name + '_vocab.txt')
        data_manager.save_txt(self.vocab.decoder)
        vector_path = data_manager.make_abs(name + '_vectors.npy')
        np.save(vector_path, self.vectors)

    def load_state(self, name):
        data_manager = IMDBDataManager(txt_path=name + '_vocab.txt')
        decoder = data_manager.load_txt()
        vector_path = data_manager.make_abs(name + '_vectors.npy')
        self.vectors = np.load(vector_path)
        self.vocab = Vocab(decoder)
        # if not self.genre_encoder.ready():
            # self.genre_encoder.fit(y)
        assert self.fitted

    @property
    def fitted(self):
        vocab_ready = self.vocab is not None and len(self.vocab.decoder)
        vectors_ready = self.vectors is not None and len(self.vectors)
        return vocab_ready and vectors_ready and self.genre_encoder.ready()

    # 5 шагов предобработки:

    def _preprocess0(self, df):
        """ Split df to X = name + text and y = genre """
        assert set(df) == {'genre', 'name', 'text'}
        return df['name'] + '.' + df['text'], df['genre']

    def _preprocess1(self, y):
        """ Encode y """
        if not self.genre_encoder.ready():
            self.genre_encoder.fit(y)
        y = y.apply(self.genre_encoder.encode)
        return y.values

    def _preprocess2(self, X, do_fit):
        """ Tokenize X """
        if do_fit:
            X = self.tokenizer.fit_transform(X)
            self.vocab, self.vectors = make_vocab(self.tokenizer.unique_tokens)
            # self.vectors = torch.from_numpy(self.vectors)
            self.tokenizer.clear()
        else:
            X = self.tokenizer.transform(X)
        return X

    def _preprocess3(self, X):
        """ Encode X """
        X = X.apply(self.vocab.encode_tokens) # Series of arrays
        assert not X.isna().any()
        return X

    def _preprocess4(self, X, y, filter_unknown=True):
        X_filtered = []
        lengths = []
        y_filtered = []
        for code_seq, target in zip(X, y):
            l = len(code_seq)
            n_unk = (code_seq == 1).sum()
            if n_unk < self.threshold * l or not filter_unknown:
                lengths.append(l)
                X_filtered.extend(code_seq.tolist())
                y_filtered.append(target)

        print(f'English filter: {1 - len(y_filtered)/len(y)} filtered')
        return np.array(X_filtered), np.array(lengths), np.array(y_filtered)

    def fit_transform(self, df):
        return self.transform(df, do_fit=True, filter_unknown=True)

    def transform(self, df, do_fit=False, filter_unknown=True):
        X, y = self._preprocess0(df)  # X=name+text, y=genre : pd.Series
        y = self._preprocess1(y)      # encoded y : np.array

        assert do_fit or self.fitted
        if do_fit and self.fitted:
            print(f"[self.__class__.__name__] Warning: refitting")

        X = self._preprocess2(X, do_fit=do_fit) # tokenized X : pd.Series
        X = self._preprocess3(X) # encoded X
        X, lengths, y = self._preprocess4(X, y, filter_unknown) # filtered X as 1d array, lengths for splitting and corresponding y
        return X, lengths, y


def preprocess(path, do_fit=False, filename=None, unk_threshold=0.1, npz_path=None):
    """ Использование предобработчика собрано в данную функцию """
    data_manager = IMDBDataManager()
    path = Path(path)
    train_mode = path.name.startswith('train') and do_fit
    if path.suffix == '.pq':
        df = data_manager.load_pq(path)
    elif path.suffix == '.csv':
        assert not train_mode
        df = data_manager.load_csv(path)
    else:
        raise ValueError(f"Wrong path extension '{path.suffix}'")

    print(f"Loaded df with colunms {list(df)}")
    if "genre" not in df:
        df["genre"] = "unset"
    if filename is None:
        filename = path.stem
    data_preprocessor = DataPreprocessor(txt_path=filename + '_genres.txt', unk_threshold=unk_threshold)
    if not train_mode:
        data_preprocessor.load_state(filename)
    filter_unknown = (path.name.startswith('train') or path.name.startswith('val'))
    X, lengths, y = data_preprocessor.transform(df, do_fit=train_mode, filter_unknown=filter_unknown)
    assert len(lengths) == len(y)
    assert len(X) == sum(lengths)
    if train_mode:
        data_preprocessor.save_state(filename)
    if npz_path is None:
        npz_path = path
    data_manager.save_xy(npz_path, X=X, lengths=lengths, y=y)
    return X, lengths, y

def main_train(train_name):
    print('train:', train_name)
    X, lengths, y = preprocess(path=train_name + '.pq', do_fit=True)
    print(X[:5])
    print(lengths[:5])
    print(y[:5])

def main_val(filename, val_name):
    print('val:', val_name)
    X, lengths, y = preprocess(path=val_name + '.pq', do_fit=False, filename=filename)
    print(X[:5])
    print(lengths[:5])
    print(y[:5])


if __name__ == '__main__':
    args = take_args(description=f'Подготовить train и val')
    val_percent, test_percent = read_conf(path=args.conf)
    train_percent = 100 - val_percent - test_percent
    train_name = f'train{train_percent}'
    val_name = f'val{val_percent}'
    main_train(train_name)
    main_val(filename=train_name, val_name=val_name)

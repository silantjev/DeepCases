from pathlib import Path
import zipfile
import numpy as np

from .imdb_data_manager import IMDBDataManager


def load_glove(path='glove.6B.300d.txt'):
    """ Загрузить векторы  """
    abs_path = IMDBDataManager().make_abs(path)
    if not abs_path.is_file() and abs_path.suffix != ".zip":
        abs_path = Path(str(abs_path) + '.zip')
    assert abs_path.is_file(), f"Neither file {path} nor {path}.zip not found"

    def parse_glove_file(file_obj, is_binary=False):
        keys = []
        vectors = []
        for line in file_obj:
            if is_binary: # zip
                line = line.decode('utf-8')
            contents = line.rstrip().split()
            key = contents[0]
            vector = np.array(contents[1:], dtype=np.float32)
            keys.append(key)
            vectors.append(vector)
        return keys, vectors

    if (abs_path.suffix == ".zip"):
        with zipfile.ZipFile(str(abs_path), 'r') as zip_ref:
            with zip_ref.open('glove.6B.300d.txt', 'r') as file:
                keys, vectors = parse_glove_file(file, is_binary=True)
    else:
        with open(str(abs_path), 'r', encoding='utf-8') as file:
            keys, vectors = parse_glove_file(file, is_binary=False)

    indices = {key: i for i, key in enumerate(keys)}
    return indices, np.array(vectors)


class Vocab:
    """ Словарь токенов для кодировки токенизированного текста
        в последовательность целых чисел и обратно
    """
    def __init__(self, decoder):
        assert '<unk>' in decoder
        self.decoder = decoder
        self.encoder = {token: i for i, token in enumerate(decoder)}
        self.unk_code = self.encoder['<unk>']

    def encode(self, token):
        return self.encoder.get(token, self.unk_code)

    def encode_tokens(self, tokens):
        return np.array(list(map(self.encode, tokens)))

    def decode(self, code):
        return self.decoder[code]

    def decode_tokens(self, encoded_list):
        return list(map(self.decode, encoded_list))


def make_vocab(tokens):
    """ Создаёт словарь токенов и массив им соответствующих векторов
        (порядок токенов в словаре согласован с порядком векторов)
        В качестве токенов берутся те, что есть в массиве tokens,
        предположительно полученного из train-данных,
        а так же есть в словаре glove, поскольку только для них имеются векторы.
        Остальные токены будут заменены токеном <unk>
    """
    assert isinstance(tokens, set)
    tokens -= {'<pad>', '<unk>'} # на всякий
    assert tokens
    decoder = ['<pad>', '<unk>'] # паддинг и «неизвестный токен»
    indices, glove_vectors = load_glove()
    n, m = glove_vectors.shape
    assert len(indices) == n
    # Векторы для <pad> и <unk>
    mean_vect = glove_vectors.mean(axis=0)
    vectors = [np.zeros([m], dtype=np.float32), mean_vect]
    for token in tokens:
        index = indices.get(token)
        if index is None:
            continue
        decoder.append(token)
        vectors.append(glove_vectors[index])

    assert len(decoder) == len(vectors)

    return Vocab(decoder), np.array(vectors)


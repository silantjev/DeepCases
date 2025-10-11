from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'

assert DATA.is_dir()

class Loader:
    """ Сохранение/загрузка данных и состояний в файлы/из фалов"""
    def __init__(self, *, data_dir=DATA, txt_path=None):
        self.data_dir = Path(data_dir) # папка где лежат файлы
        self.txt_path = txt_path # путь файла для сохранения/загрузки жанров или других текстовых данных
        if self.txt_path:
            self.txt_path = self.make_abs(self.txt_path)

    def make_abs(self, path):
        """ Если дано просто имя фала, то перевести его в путь в заданной папке """
        path = Path(path)
        if not path.is_absolute():
            path = self.data_dir / path
        return path

    def load_csv(self, path):
        path = self.make_abs(path)
        if not path.is_file():
            return
        return pd.read_csv(path, encoding='utf-8')

    def load_pq(self, path):
        path = self.make_abs(path)
        if not path.is_file():
            return
        return pd.read_parquet(path)

    def load_txt(self):
        assert self.txt_path is not None
        if not self.txt_path.is_file():
            return []
        with open(str(self.txt_path), 'r', encoding='utf-8') as f:
            first_line = f.readline()
        while first_line.endswith('\n'):
            first_line = first_line[:-1]
        return first_line.split(' ')

    def save_txt(self, genres):
        assert self.txt_path is not None
        assert self.txt_path.parent.is_dir()
        assert self.txt_path.name.startswith('train')
        with open(str(self.txt_path), 'w', encoding='utf-8') as f:
            f.write(' '.join(genres) + '\n')

    def save_pq(self, df, path):
        path = self.make_abs(path)
        df.to_parquet(path)

    def save_xy(self, path, **kwargs):
        path = self.make_abs(path).with_suffix('.npz')
        np.savez_compressed(path, **kwargs)

    def load_xy(self, path, keys=['X', 'lengths', 'y']):
        path = self.make_abs(path).with_suffix('.npz')
        with np.load(path) as l:
            return [l[k] for k in keys]

# Отдельные функции для сохранения/загрузки npz-данных
def save_npz(path, dict_to_save):
    for k, v in dict_to_save.items():
        dict_to_save[k] = np.array(v)
    path = Path(path).with_suffix('.npz')
    np.savez_compressed(path, **dict_to_save)

def load_npz(path):
    with np.load(path) as data:
        return dict(data)

import json
from pathlib import Path
import scipy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]

# Данные лежат в фиксированном месте:
DATA = ROOT / 'data'
FLOWERS_DIR = DATA / 'flowers-102' 
IMAGE_DIR = FLOWERS_DIR / 'jpg'
assert IMAGE_DIR.is_dir()


class FlowerDataManager:
    """ Загрузка первичных данных:
            - имена файлов с изображениями
            - метки классов в порядке соответствующем именам фалов
            - название классов
        Разделение данных на train/val/test:
            делится не массив с именами файлов, а массив индексов;
            далее можно извлекать имена нужных фалов из массива имён paths,
            обращаясь по индексам из массивов train_idx, val_idx, test_idx;
            можно так же воспользоваться массивами меток train_labels, val_labels, test_labels.
        Подсчёт баланса классов для проверки их сбалансированности
        Загрузка происходит в конструкторе,
            разделение вызывается методом split,
            баланс считается при вызове геттера
    """
    def __init__(self):
        self.imagelabels = self._load_image_labels()
        assert self.imagelabels.dtype == np.uint8
        self.total_images = len(self.imagelabels)
        self.flower_classes = self._load_class_names()
        self.num_classes = len(self.flower_classes)
        self.paths = self._load_image_paths()
        self.all_indices = np.arange(self.total_images)

    def _load_image_labels(self):
        imagelabels = scipy.io.loadmat(FLOWERS_DIR / 'imagelabels.mat')['labels'][0]
        return imagelabels - 1 # от нуля

    def _load_class_names(self):
        with open(DATA / 'cat_to_name.json') as f:
            flower_classes = json.load(f)
        return {int(label) - 1: name for label, name in flower_classes.items()}

    def _load_image_paths(self):
        assert hasattr(self, 'imagelabels')
        assert hasattr(self, 'flower_classes')
        paths = []
        label_set = set()
        for i, label in enumerate(self.imagelabels):
            filepath = IMAGE_DIR / f'image_{i+1:05d}.jpg'
            label_set.add(label)
            assert filepath.exists(), filepath
            paths.append(filepath)
        assert label_set == set(self.flower_classes)
        assert len(paths) == self.total_images
        return paths

    def _count_labels(self, labels):
        counter = np.zeros(self.num_classes)
        for label in labels:
            counter[label] += 1
        return counter

    def _count(self):
        train_counter = self._count_labels(self.train_labels)
        val_counter = self._count_labels(self.val_labels)
        test_counter = self._count_labels(self.test_labels)
        self.counter_df = pd.DataFrame({'train': train_counter, 'val': val_counter, 'test': test_counter}).T.astype(int)
        self.counter_df_norm = self.counter_df / self.counter_df.apply(np.sum, axis=0)

    def split(self, test_size=0.2, val_size=0.16):
        assert 0 < test_size + val_size < 1
        assert hasattr(self, 'imagelabels')
        self.train_idx, self.test_idx, self.train_labels, self.test_labels = train_test_split(
            self.all_indices, self.imagelabels, test_size=test_size, random_state=42, stratify=self.imagelabels)
        self.train_idx, self.val_idx, self.train_labels, self.val_labels = train_test_split(
            self.train_idx, self.train_labels, test_size=val_size / (1-test_size), random_state=42, stratify=self.train_labels)
        return self.get_data()

    # Геттеры для удобства:

    def get_data(self):
        assert hasattr(self, 'train_labels')
        return self.train_idx, self.train_labels, self.val_idx, self.val_labels, self.test_idx, self.test_labels

    def get_counter(self, normalized=True):
        if not hasattr(self, 'train_labels'):
            self.split()
        if not hasattr(self, 'train_counter'):
            self._count()
        if normalized:
            return self.counter_df_norm
        return self.counter_df

    def get_paths(self):
        return self.paths


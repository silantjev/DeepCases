from .load_data import Loader

class GenreEncoder:
    """ Кодировщик классов, в данном случае классы — жанры фильмов """
    def __init__(self, txt_path):
        self.loader = Loader(txt_path=txt_path)
        self.genres = self.loader.load_txt() # если файла ещё нет, то возвращается пустой список
        if self.genres:
            self._make_codes()

    def _make_codes(self):
        assert not hasattr(self, "len")
        self.len = len(self.genres)
        # Создаём словарь для кодирования известных жанров
        self.codes = {genre: i for i, genre in enumerate(self.genres)}
        # Добавляем элемент для неизвестных жанров
        self.genres.append("<other>")

    def ready(self):
        return bool(self.genres) and hasattr(self, "len")

    def fit(self, train_y):
        """ Извлекаем жанры из тренировочных данных """
        self.genres = list(train_y.unique())
        self.loader.save_txt(self.genres)
        self._make_codes()

    def encode(self, genre):
        return self.codes.get(genre, self.len)

    def decode(self, code):
        return self.genres[code]


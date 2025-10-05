from pathlib import Path
import contractions
import nltk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
while (PROJECT_ROOT.name not in ['', '.', 'home', 'DeepCases']):
    PROJECT_ROOT = PROJECT_ROOT.parent

assert PROJECT_ROOT.name == 'DeepCases'

# download nltk data to custom path
NLTK_DATA_DIR = PROJECT_ROOT / 'nltk_data'
NLTK_DATA_DIR.mkdir(exist_ok=True)
assert NLTK_DATA_DIR.is_dir()
#nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
#nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)  # tokinizer
nltk.download('punkt', download_dir=NLTK_DATA_DIR, quiet=True)  # for the tokinizer
nltk.download('punkt_tab', download_dir=NLTK_DATA_DIR, quiet=True)  # for the tokinizer
# add custum path
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_DIR)


def tokenize(text):
    expanded_text = contractions.fix(text) # Заменяем сокращения
    lower_text = expanded_text.lower()
    tokens = nltk.tokenize.word_tokenize(lower_text)
    return tokens


class Tokenizer:
    """
    Кастомный токенизатор, который:
    1. Заменяет сокращения на полные формы (I'm → I am)
    2. Приводит текст к нижнему регистру
    3. Токенизирует, сохраняя пунктуацию
    4. Не удаляет стоп-слова
    5. Не лемматизирует
    """
    def __init__(self):
        self.unique_tokens = set()

    def _tokenize_with_fit(self, text):
        tokens = tokenize(text)
        self.unique_tokens.update(tokens)
        return tokens
        # return ' '.join(tokens)

    def _just_tokenize(self, text):
        tokens = tokenize(text)
        return tokens
        # return ' '.join(tokens)

    def fit_transform(self, series):
        return series.apply(self._tokenize_with_fit)

    def transform(self, series):
        return series.apply(self._just_tokenize)

    def clear(self):
        self.unique_tokens = set()


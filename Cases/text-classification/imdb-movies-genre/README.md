# Классификация: датасет imdb-movies-genre-classification

Имеются данные о фильме:
 - Название
 - Дата выхода
 - Описание

Требуется определить жанр из 27 категорий.

## Загрузка данных

Скачать файл imdb-movies-genre-data.zip по ссылке https://drive.google.com/file/d/1vz4mSSIQl8mXSDBZ_OXfhbN0T1P1gHdr/view?usp=drive_link
в папку data. Распаковать:
```bash
cd data
unzip imdb-movies-genre-data.zip
rm imdb-movies-genre-data.zip
```
Появятся 2 файла:
 - train.csv — размеченные данные
 - test.csv — не размеченные данные

## Загрузка предобученных эмбеддингов glove.6B.300d

Запистить скрипт (требует wget)
```bash
./fetch_glove.sh
```

Либо вручную скачать файл glove.6B.300d.txt по адресу https://www.kaggle.com/datasets/thanakomsn/glove6b300dtxt или https://huggingface.co/spaces/Vrk/SkimLit/blob/main/glove.6B.300d.txt и положить в папку data. Можно заархивировать для экономии места на диске:
```bash
cd data
zip glove.6B.300d.txt.zip glove.6B.300d.txt
rm glove.6B.300d.txt
```

## Установка зависимостей

Рекомендуется использовать Python 3.11 и виртуальное окружение

На ubuntu:
```bash
sudo apt-get update
sudo apt-get install python3.11
python3.11 -m venv ".venv"
source ".venv/bin/activate"
```

Также необходимо установить модули:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
Установить нужный PyTorch
```bash
pip3 install torch --index-url https://download.pytorch.org/whl/$xxx
```
где xxx=cpu или cu128 (указать куду нужной версии), см. https://docs.pytorch.org/get-started/locally

Автоматически (ubuntu):
```bash
rm -r .venv
sudo apt-get update
./install_venv.sh
```
В последней команде можно указать xxx, например:
```bash
./install_venv.sh cpu
```

## Использование common

В коде используются общие модули из папки пакета `common`, лежащего в корне проекта.
Чтобы работали импорты из `common`, необходимы символьные ссылки:
```bash
cd src
ln -s ../../../../common common
cd ..
```

Либо можно добавить корень проекта в переменную PYTHONPATH
```bash
export PYTHONPATH=$(realpath ../../..)${PYTHONPATH+:$PYTHONPATH}
```

## Подготовка данных

Разделение train.csv на train и val в пропорции 3:1
```bash
./run_split_data.sh
```
Либо используйте скрипт `src/split_data.py`

Предобработка данных:
```bash
./run_prepare_data.sh
```
Либо используйте скрипт `src/prepare_data.py`

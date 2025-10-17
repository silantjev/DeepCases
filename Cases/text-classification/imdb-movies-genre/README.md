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
pip install torch --index-url https://download.pytorch.org/whl/$xxx
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

На windows (git bash):
```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
xxx=cpu
pip install torch --index-url https://download.pytorch.org/whl/$xxx
```

## Использование common

В коде используются общие модули из папки пакета `common`, лежащего в корне проекта.
Чтобы работали импорты из `common`, необходимы символьные ссылки:
```bash
cd src
ln -s ../../../../common common
cd ..
```

Либо (в т.ч. на windows) можно добавить корень проекта в переменную PYTHONPATH (bash/git bash):
```bash
export PYTHONPATH=$(realpath ../../..)${PYTHONPATH+:$PYTHONPATH}
```

## Подготовка данных

### Разделение

Отредактируйте файл `SPLIT_CONF.json` (используется по умолчанию) или создайте свой такого же формата, задав в нём параметры для разделения данных:
 - `val\_percent` — размер данных для валидации в процентах
 - `test\_percent` — размер данных для итоговой оценки в процентах

Разделение данных (по умолчанию train.csv) на train/val/test
```bash
python src/split_data.py [-h] [--data DATA] [--conf CONF]
```
где DATA — csv-файл с данными, CONF — конфигурационный json-файл с процентами

На linux (в bash) можно воспользоваться скриптом, который автоматически активирует виртуальное окружение:
```bash
./run_split_data.sh [-h] [--data DATA] [--conf CONF]
```

### Предобработка данных:

```bash
python src/prepare_data.py [-h] [--conf CONF]
```
или
```bash
./run_prepare_data.sh [-h] [--conf CONF]
```
где CONF — ровно тот же json-файл.

## Обучение

Задайте модель и параметры обучения в файле `train_conf.yaml` (используется по умолчанию) или в другом yaml-файле.

Запустить обучение:
```bash
python main.py [-h] [--params PARAMS] [--conf CONF]
```
или
```bash
./run_model_training.sh [-h] [--params PARAMS] [--conf CONF]
```
где CONF — тот же json-файл с процентами, а PARAMS — конфигурационный yaml-файл


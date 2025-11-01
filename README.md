# DeepCases

Случаи глубокого обучения использующие общий код из папки `common`

## Установка зависимостей

Рекомендуется использовать Python 3.11 и виртуальное окружение

На ubuntu:
---------

```bash
sudo apt-get update
sudo apt-get install python3.11 python3.11-venv
```
Если нужной версии python нет на ubuntu, то сначала подключить репозиторий
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
```

Виртуальное окружение (через `pip`):
```bash
python3.11 -m venv ".venv"
source .venv/bin/activate
```

Также необходимо установить модули.
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
Установить нужный PyTorch
```bash
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/$xxx
```
где xxx=cpu или cu128 (указать куду нужной версии или опустить --incex-url для cuda по умолчанию), см. https://docs.pytorch.org/get-started/locally

Через `uv`:
```
curl -LsSf https://astral.sh/uv/install.sh | sh # установка uv, если ещё нет
uv venv --python /usr/bin/python3.11
source .venv/bin/activate
uv sync --extra torch-$xxx
```

`Автоматически`:
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



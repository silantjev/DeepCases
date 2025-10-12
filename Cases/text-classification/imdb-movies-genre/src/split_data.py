import argparse
import json

# Импорт изобщего кода
from common.data_preparation.split_dataset import split_df
from common.utils import find_file

# Локальный импорт
from utils.load_data import Loader, ROOT

# Делим train.csv на train и val, сохраняя результат в parquet-файлы

DEFAULT_PATH = ROOT / 'default_split_conf.json'

def write_default_conf(path=DEFAULT_PATH):
    default_values = {'val_percent': 25, 'test_percent': 0}
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(default_values, f, indent=4)
    print(f"Default config '%s' created" % path)
    return default_values['val_percent'], default_values['test_percent']

def read_conf(path=None, create_default=True):
    if path is None:
        path = DEFAULT_PATH
        if not path.exists() and create_default:
            return write_default_conf(path)

    path = find_file(path, root=ROOT)
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    val_percent = data.get('val_percent')
    assert val_percent is not None, f"Field 'val_percent' not found in config file \"{path}\""
    assert isinstance(val_percent, int) and 0 < val_percent < 100, "Field 'val_percent' should be integer in (0, 100)"
    test_percent = data.get('test_percent', 0)
    assert isinstance(test_percent, int) and 0 <= test_percent < 100, f"Field 'test_percent' should be integer in [0, 100)"

    return val_percent, test_percent


def split_and_save(path='train.csv', val_percent=25, test_percent=0):
    loader = Loader()
    def save_part_df(df, name, percent):
        path=f'{name}{percent}.pq'
        loader.save_pq(df, path=path)
        print(f"dataset \'{name}\' saved to \"{path}\"")

    df = loader.load_csv(path)
    train_df, val_df, test_df = split_df(df, val_percent=val_percent, test_percent=test_percent, target='genre')


    save_part_df(train_df, name='train', percent=100 - val_percent - test_percent)

    save_part_df(val_df, name='val', percent=val_percent)

    if test_percent:
        assert test_df is not None
        save_part_df(test_df, name='test', percent=test_percent)

def take_args(default_data_path='train.csv'):
    parser = argparse.ArgumentParser(description=f'Разделить данные на train/val/test', add_help=False) 

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='показать справку и выйти')
    parser.add_argument('--data', type=str, default=default_data_path, help='csv-файл с данными')
    parser.add_argument('--conf', type=str, default=None, help='конфигурационный файл с процентами')

    return parser.parse_args()

if __name__ == '__main__':
    args = take_args()
    val_percent, test_percent = read_conf(path=args.conf)
    print(f"{val_percent=}, {test_percent=}")
    split_and_save(path=args.data, val_percent=val_percent, test_percent=test_percent)

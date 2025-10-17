import argparse

# Импорт изобщего кода
from common import split_dataset, read_conf

# Локальный импорт
from utils.imdb_data_manager import IMDBDataManager, ROOT

# Делим train.csv на train и val, сохраняя результат в parquet-файлы


def split_and_save(path='train.csv', val_percent=25, test_percent=0):
    data_manager = IMDBDataManager()
    def save_part_df(df, name, percent):
        path=f'{name}{percent}.pq'
        data_manager.save_pq(df, path=path)
        print(f"dataset \'{name}\' saved to \"{path}\"")

    df = data_manager.load_csv(path)
    train_df, val_df, test_df = split_dataset.split_df(df, val_percent=val_percent, test_percent=test_percent, target='genre')


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
    val_percent, test_percent = read_conf(root=ROOT, path=args.conf)
    print(f"{val_percent=}, {test_percent=}")
    split_and_save(path=args.data, val_percent=val_percent, test_percent=test_percent)

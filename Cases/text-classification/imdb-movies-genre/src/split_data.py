# Импорт из общего кода
from common.data_preparation.split_dataset import split_df

# Локальный импорт
from utils.load_data import Loader, ROOT

# Делим train.csv на train и val, сохраняя результат в parquet-файлы


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

if __name__ == '__main__':
    split_and_save(path='train.csv')

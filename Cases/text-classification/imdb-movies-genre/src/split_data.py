from sklearn.model_selection import train_test_split

from utils.load_data import Loader

# Делим train.csv на train и val, сохраняя результат в parquet-файлы

def split_train(path='train.csv', val_percent=25):
    loader = Loader()
    df = loader.load_csv(path)
    train_df, val_df = train_test_split(df, test_size=val_percent / 100, random_state=42, stratify=df['genre'])
    train_path=f'train{100-val_percent}.pq'
    loader.save_pq(train_df, path=train_path)
    print(f"Train dataset saved to \"{train_path}\"")
    val_path = f'val{val_percent}.pq'
    loader.save_pq(val_df, path=val_path)
    print(f"Validation dataset saved to \"{val_path}\"")

if __name__ == '__main__':
    split_train()

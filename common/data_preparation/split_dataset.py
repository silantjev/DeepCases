from sklearn.model_selection import train_test_split

def split_df(df, val_percent, test_percent=0, target=None):
    assert 0 < val_percent < 100
    assert 0 <= test_percent < 100
    assert val_percent + test_percent < 100

    if target is None:
        stratify=None
    else:
        stratify = df[target]

    train_df, val_df = train_test_split(df, test_size=val_percent / 100, random_state=42, stratify=stratify)

    if test_percent == 0:
        return train_df, val_df, None

    train_df, test_df, _ = split_df(train_df, val_percent=test_percent / (100 - val_percent), target=target)
    return train_df, val_df, test_df


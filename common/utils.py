from pathlib import Path
import numpy as np

def find_file(path, root=None):
    path = Path(path)
    if not path.is_absolute():
        if path.exists():
            path = path.resolve()
        elif root is not None:
            path = Path(root) / str(path)
    if path.is_dir():
        raise FileNotFoundError(f"File \"{path}\" is a folder")
    if not path.exists():
        raise FileNotFoundError(f"File \"{path}\" not found")
    return path


# Отдельные функции для сохранения/загрузки npz-данных
def save_npz(path, dict_to_save):
    for k, v in dict_to_save.items():
        dict_to_save[k] = np.array(v)
    path = Path(path).with_suffix('.npz')
    np.savez_compressed(path, **dict_to_save)

def load_npz(path):
    with np.load(path) as data:
        return dict(data)

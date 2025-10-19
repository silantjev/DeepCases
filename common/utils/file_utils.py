from pathlib import Path
import numpy as np

def find_file(path, root=None, dir_allowed=False, file_allowed=True):
    """ Ищет файл
            - path: путь абсолютный или относительный
            - root: директория: если path относительный, то искать в директории root
            - dir_allowed: это может быть директория
            - file_allowed: это может быть файл 
    """
    assert dir_allowed or file_allowed
    path = Path(path)
    if not path.is_absolute():
        if path.exists():
            path = path.resolve()
        elif root is not None:
            path = Path(root) / str(path)
    if not path.exists():
        if file_allowed and file_allowed:
            file_or_dir = "File or folder"
        elif file_allowed:
            file_or_dir += "File"
        else:
            file_or_dir = "Folder"
        raise FileNotFoundError(f"{file_or_dir} \"{path}\" not found")
    if not dir_allowed and path.is_dir():
        raise FileExistsError(f"File \"{path}\" is a folder")
    if not file_allowed and path.is_file():
        raise FileExistsError(f"File \"{path}\" is not a folder")
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

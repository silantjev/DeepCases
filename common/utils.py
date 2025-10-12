from pathlib import Path

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


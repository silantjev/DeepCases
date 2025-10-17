import json

# Импорт изобщего кода
from .. import find_file

DEFAULT_FILENAME = 'SPLIT_CONF.json'

def write_default_conf(path):
    assert path.parent.is_dir()
    default_values = {'val_percent': 20, 'test_percent': 10}
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(default_values, f, indent=4)
    print(f"Default config '%s' created" % path)
    return default_values['val_percent'], default_values['test_percent']

def read_conf(root, path=None, create_default=True):
    assert root.is_dir(), f"{root} is not directory"
    if path is None:
        path = root / DEFAULT_FILENAME
        if not path.exists() and create_default:
            return write_default_conf(path)

    path = find_file(path, root=root)
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    val_percent = data.get('val_percent')
    assert val_percent is not None, f"Field 'val_percent' not found in config file \"{path}\""
    assert isinstance(val_percent, int) and 0 < val_percent < 100, "Field 'val_percent' should be integer in (0, 100)"
    test_percent = data.get('test_percent', 0)
    assert isinstance(test_percent, int) and 0 <= test_percent < 100, f"Field 'test_percent' should be integer in [0, 100)"

    return val_percent, test_percent


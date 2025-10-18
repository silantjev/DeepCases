import sys
import yaml
from pydantic import ValidationError

def read_yaml(yaml_path, Config):
    config_dict = None
    try:
        with open(yaml_path, "r", encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        if config_dict is None:
            print(f"[Error]: Empty config \"{yaml_path}\"")
    except FileNotFoundError as e:
        print(f"[Error]: File \"{yaml_path}\" not found:\n\t{e}")
    except PermissionError as e:
        print(f"[Error]: No permissions for {yaml_path}:\n\t{e}")
    except UnicodeDecodeError as e:
        print(f"[Error]: Encoding failed:\n\t{e}")
    except yaml.YAMLError as e:
        print(f"[Error]: YAML syntax:\n\t{e}")
    except OSError as e:
        print(f"[Error]: System error:\n\t{e}")

    if config_dict is None:
        return None

    if not isinstance(config_dict, dict):
        print(f"[Error] Expected dict in YAML config, got {type(config_dict)}")
        return None

    try:
        config = Config(**config_dict)
    except (ValidationError, AssertionError) as exc:
        print("[Error]: failed to parse yaml-file.")
        print("Validation errors:\n")
        if isinstance(exc, ValidationError):
            print(exc.json(indent=2))
        else:
            print(exc)
        print(f"\nCheck your configuration yaml-file \"{yaml_path}\"")
        print(f"You can find a sample in the directory \"Config_samples\"")
        return None

    return config

def save_yaml(path, data):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(
                    data,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                    indent=2,
                )

    except OSError as e:
        print(f"[Error]: System error while saving YAML file:\n\t{e}")
        return False
    except yaml.YAMLError as e:
        print(f"[Error] YAML serialization failed:\n\t{e}")
        return False
    except TypeError as e:
        print(f"[Error] Invalid data type encountered during YAML serialization:\n\t{e}")
        return False

    return True

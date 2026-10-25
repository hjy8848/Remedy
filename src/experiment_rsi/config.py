from pathlib import Path

import yaml


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    for section, key in (("model", "path"), ("alfworld", "config_path")):
        value = config[section][key]
        if not value:
            raise ValueError("{}.{} must be set in the run config".format(section, key))

    return config

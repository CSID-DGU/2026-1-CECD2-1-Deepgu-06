import json
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pipeline_config(config_path=None):
    config_file = Path(config_path or "/home/deepgu/test/configs/pipeline_config.json")
    return load_json(config_file)


def load_label_map(label_map_path=None):
    label_file = Path(label_map_path or "/home/deepgu/test/configs/label_map.json")
    return load_json(label_file)

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"


def load_config(config_path=None):
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

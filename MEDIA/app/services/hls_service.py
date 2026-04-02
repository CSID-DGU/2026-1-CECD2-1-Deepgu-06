from pathlib import Path
from typing import Iterable


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def ensure_dirs(paths: Iterable[str]) -> None:
    for path in paths:
        ensure_dir(path)
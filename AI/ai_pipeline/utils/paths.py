from pathlib import Path


TEST_ROOT = Path("/home/deepgu/test")


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

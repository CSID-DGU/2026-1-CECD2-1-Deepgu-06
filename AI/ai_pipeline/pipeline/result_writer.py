import json
from pathlib import Path

from utils.paths import ensure_dir


def write_results(results_path, payload):
    results_path = Path(results_path)
    ensure_dir(results_path.parent)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_debug(debug_dir, clip_results):
    debug_dir = ensure_dir(debug_dir)
    debug_path = debug_dir / "clip_results.json"
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(clip_results, f, indent=2, ensure_ascii=False)

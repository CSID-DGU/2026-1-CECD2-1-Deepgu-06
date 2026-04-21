from pathlib import Path


def load_prompt(prompt_path=None):
    prompt_file = Path(prompt_path or "/home/deepgu/test/configs/prompts/vlm_fight_fall.txt")
    return prompt_file.read_text(encoding="utf-8")

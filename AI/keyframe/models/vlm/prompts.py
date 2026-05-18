from pathlib import Path

_DEFAULT_PROMPT = Path(__file__).parent / "prompts" / "vlm_anomaly.txt"


def load_prompt(prompt_path=None):
    prompt_file = Path(prompt_path) if prompt_path else _DEFAULT_PROMPT
    return prompt_file.read_text(encoding="utf-8")

import json
import re


LABEL_ALIASES = {
    "anomaly": "uncertain",
    "violent": "fight",
    "violence": "fight",
    "fighting": "fight",
    "attack": "fight",
    "falling": "fall"
}

VALID_LABELS = {"fight", "fall", "normal", "uncertain"}


def parse_vlm_output(output):
    try:
        return json.loads(output)
    except Exception:
        match = re.search(r"\{.*\}", output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass

    return {
        "label": "uncertain",
        "confidence": 0.0,
        "evidence": output.strip()
    }


def normalize_vlm_output(result):
    label = str(result.get("label", "uncertain")).strip().lower()
    label = LABEL_ALIASES.get(label, label)
    if label not in VALID_LABELS:
        label = "uncertain"

    confidence = result.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    evidence = str(result.get("evidence") or result.get("description") or "").strip()

    return {
        "label": label,
        "confidence": confidence,
        "evidence": evidence
    }

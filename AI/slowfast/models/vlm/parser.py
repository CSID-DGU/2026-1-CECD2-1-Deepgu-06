import json
import re


def _extract_json_payload(text):
    text = str(text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _normalize_label(label):
    lowered = str(label).strip().lower()
    if lowered in {"fight", "fighting", "violent", "violence", "assault", "yes", "true"}:
        return "fight"
    return "non_fight"


def _extract_partial_fields(text):
    text = str(text)
    label_match = re.search(r'"label"\s*:\s*"(fight|non_fight)"', text, flags=re.IGNORECASE)
    confidence_match = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', text, flags=re.IGNORECASE)
    reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)', text, flags=re.IGNORECASE | re.DOTALL)

    if not label_match and not confidence_match:
        return None

    payload = {
        "label": _normalize_label(label_match.group(1) if label_match else "non_fight"),
        "confidence": float(confidence_match.group(1)) if confidence_match else 0.5,
        "reasoning": reasoning_match.group(1).strip() if reasoning_match else text,
    }
    return payload


def parse_vlm_response(text):
    if isinstance(text, dict):
        payload = dict(text)
        payload["label"] = _normalize_label(payload.get("label", "non_fight"))
        payload["confidence"] = float(payload.get("confidence", 0.5))
        payload.setdefault("reasoning", "")
        return payload

    candidate = _extract_json_payload(text)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        partial = _extract_partial_fields(text)
        if partial is not None:
            return partial
        lowered = str(text).lower()
        label = "fight" if "fight" in lowered or "violence" in lowered or "assault" in lowered else "non_fight"
        confidence = 0.7 if label == "fight" else 0.3
        return {"label": label, "confidence": confidence, "reasoning": text}

    payload["label"] = _normalize_label(payload.get("label", "non_fight"))
    payload["confidence"] = float(payload.get("confidence", 0.5))
    payload.setdefault("reasoning", "")
    return payload

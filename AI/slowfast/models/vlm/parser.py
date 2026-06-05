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
    scene_match = re.search(r'"scene_description"\s*:\s*"([^"]*)', text, flags=re.IGNORECASE | re.DOTALL)

    if not label_match and not confidence_match:
        return None

    payload = {
        "label": _normalize_label(label_match.group(1) if label_match else "non_fight"),
        "confidence": float(confidence_match.group(1)) if confidence_match else 0.5,
        "scene_description": scene_match.group(1).strip() if scene_match else "",
        "reasoning": reasoning_match.group(1).strip() if reasoning_match else text,
    }
    return payload


def parse_vlm_response_3label(text):
    """3-label 응답 파싱: level(0/1/2), confidence, reasoning."""
    if isinstance(text, dict):
        payload = dict(text)
        try:
            level = int(payload.get("level", 1))
        except (TypeError, ValueError):
            level = 1
        level = max(0, min(2, level))
        return {
            "level": level,
            "confidence": float(payload.get("confidence", 0.5)),
            "reasoning": str(payload.get("reasoning", "")),
        }

    candidate = _extract_json_payload(text)
    try:
        payload = json.loads(candidate)
        level = int(payload.get("level", 1))
        level = max(0, min(2, level))
        return {
            "level": level,
            "confidence": float(payload.get("confidence", 0.5)),
            "reasoning": str(payload.get("reasoning", "")),
        }
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # 정수 level 직접 탐색
    level_match = re.search(r'"level"\s*:\s*([012])', text)
    if level_match:
        level = int(level_match.group(1))
        confidence_match = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', text)
        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)', text)
        return {
            "level": level,
            "confidence": float(confidence_match.group(1)) if confidence_match else 0.5,
            "reasoning": reasoning_match.group(1).strip() if reasoning_match else text,
        }

    # 최후 fallback: 텍스트에서 fight 단서로 추정
    lowered = str(text).lower()
    if "level 2" in lowered or ("fight" in lowered and "non" not in lowered):
        level = 2
    elif "level 0" in lowered or "normal" in lowered:
        level = 0
    else:
        level = 1
    return {"level": level, "confidence": 0.5, "reasoning": text[:100]}


def parse_vlm_response(text):
    if isinstance(text, dict):
        payload = dict(text)
        payload["label"] = _normalize_label(payload.get("label", "non_fight"))
        payload["confidence"] = float(payload.get("confidence", 0.5))
        payload.setdefault("reasoning", "")
        payload.setdefault("scene_description", "")
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
    payload.setdefault("scene_description", "")
    return payload

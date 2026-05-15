"""
Phase 1 (Temporal Clustering + Motion) + InternVL2로 pseudo-label을 생성합니다.

동작:
  1. candidate clip에 Phase 1 sampler 적용 → 대표 frame 선택
  2. 선택된 frame을 InternVL2에 입력 → anomaly/normal 판별
  3. InternVL2 응답을 파싱해 label(0/1) 저장
  4. features를 .npy로 저장 (train_frame_selector.py에서 로드)

출력:
  - pseudo_labels.json : [{clip_id, label, features_path}, ...]
  - features/{clip_id}.npy : ResNet-50 features (T, 2048)

사용법:
  cd AI/ai_pipeline
  python scripts/generate_pseudo_labels.py \
      --output_dir outputs/pseudo_labels \
      --n_frames 8

  (candidate 데이터는 main_pipeline.py의 filter_clips 결과를 직접 넘겨받는 구조로
   실제 사용 시 main_pipeline.py와 연동 필요)
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.sampler import KeyframeSampler
from models.vlm.inference import InternVL


def parse_vlm_response(response: str) -> int:
    """
    InternVL2의 JSON 응답에서 label을 파싱합니다.
    응답 형식: {"label": "anomaly" or "normal", "description": "..."}

    파싱 실패 시 키워드 기반 fallback 사용.
    반환: 1 (anomaly) or 0 (normal)
    """
    try:
        # JSON 파싱 시도
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end != 0:
            data = json.loads(response[start:end])
            label_str = data.get("label", "").lower()
            if "anomaly" in label_str:
                return 1
            if "normal" in label_str:
                return 0
    except (json.JSONDecodeError, KeyError):
        pass

    # fallback: 키워드 기반
    response_lower = response.lower()
    anomaly_keywords = ["anomaly", "abnormal", "fight", "violence", "assault", "abuse"]
    if any(w in response_lower for w in anomaly_keywords):
        return 1
    return 0


def generate(candidates, output_dir, n_frames=8):
    """
    candidates: list of dicts
        {
            "clip_id"  : str or int,
            "clip"     : list of np.ndarray (BGR frames),
            "features" : np.ndarray (T, 2048),
        }
    output_dir: 결과를 저장할 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)
    feat_dir = os.path.join(output_dir, "features")
    os.makedirs(feat_dir, exist_ok=True)

    sampler = KeyframeSampler(n_frames=n_frames)   # Phase 1
    vlm = InternVL()

    pseudo_labels = []
    failed = []

    for i, c in enumerate(candidates):
        clip_id = c["clip_id"]
        features = np.asarray(c["features"], dtype=np.float32)

        # Phase 1: frame 선택
        sampler.sample(c)
        frames = c["sampled_frames"]

        # InternVL2 추론
        try:
            response = vlm.predict(frames)
            label = parse_vlm_response(response)
        except Exception as e:
            print(f"[{i}] clip_id={clip_id} VLM 실패: {e}")
            failed.append(clip_id)
            continue

        # features 저장 (.npy)
        feat_path = os.path.join(feat_dir, f"{clip_id}.npy")
        np.save(feat_path, features)

        pseudo_labels.append({
            "clip_id"       : clip_id,
            "label"         : label,           # 0: normal, 1: anomaly
            "vlm_response"  : response,        # 디버깅용 원문
            "features_path" : feat_path,
            "selected_indices": c["selected_indices"],
        })

        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(candidates)}] 완료, anomaly={sum(x['label'] for x in pseudo_labels)}")

    # 저장
    out_path = os.path.join(output_dir, "pseudo_labels.json")
    with open(out_path, "w") as f:
        json.dump(pseudo_labels, f, indent=2, ensure_ascii=False)

    print(f"\n완료: {len(pseudo_labels)}개 저장 → {out_path}")
    print(f"  anomaly: {sum(x['label'] for x in pseudo_labels)}")
    print(f"  normal : {sum(1 - x['label'] for x in pseudo_labels)}")
    if failed:
        print(f"  실패   : {len(failed)}개 ({failed})")

    return pseudo_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/pseudo_labels")
    parser.add_argument("--n_frames", type=int, default=8)
    args = parser.parse_args()

    # 실제 사용 시: main_pipeline.py의 candidates를 여기에 넘기거나
    # 저장된 candidates를 로드해서 사용
    print("candidates를 직접 넘겨주세요. (main_pipeline.py 연동 필요)")
    print("예시: from scripts.generate_pseudo_labels import generate")
    print("      generate(candidates, args.output_dir, args.n_frames)")

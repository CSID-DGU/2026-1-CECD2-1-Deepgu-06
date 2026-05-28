from __future__ import annotations

import base64
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import cv2
import numpy as np

from models.vlm.parser import parse_vlm_response
from models.vlm.prompts import build_binary_fight_prompt, build_event_fight_prompt


class MockVLMProvider:
    def invoke(self, clip_record, prompt):
        score = float(clip_record["fighting_prob"])
        if clip_record["uncertainty"] >= 0.3:
            score = min(1.0, score + 0.1)
        label = "fight" if score >= 0.5 else "non_fight"
        confidence = score if label == "fight" else 1.0 - score
        return {
            "label": label,
            "confidence": confidence,
            "reasoning": "Mock VLM response for pipeline integration.",
        }


class InternVLVLMProvider:
    _runtime_cache: Dict[str, tuple] = {}

    def __init__(self, config):
        self.model_path = str(config.get("model_path", "OpenGVLab/InternVL2-8B"))
        self.repo_path = str(config.get("repo_path", "/home/deepgu/VERA/InternVL/internvl_chat"))
        self.sampled_frames = int(config.get("sampled_frames", 6))
        self.frame_image_size = int(config.get("frame_image_size", 448))
        self.device = str(config.get("device", "auto"))
        self.torch_dtype_name = str(config.get("torch_dtype", "bfloat16")).lower()
        self.max_new_tokens = int(config.get("max_new_tokens", 256))
        self.do_sample = bool(config.get("do_sample", False))
        self.temperature = float(config.get("temperature", 0.0))
        self.top_p = float(config.get("top_p", 0.9))
        self.verbose = bool(config.get("verbose", False))
        self.load_in_8bit = bool(config.get("load_in_8bit", False))
        self.load_in_4bit = bool(config.get("load_in_4bit", False))

    def invoke(self, clip_record, prompt):
        runtime = self._load_runtime()
        torch = runtime["torch"]
        model = runtime["model"]
        tokenizer = runtime["tokenizer"]

        sampled_frames = _sample_frames(clip_record["frames"], self.sampled_frames)
        question = self._build_question(prompt, num_frames=len(sampled_frames))
        pixel_values, num_patches_list = self._build_pixel_values(sampled_frames, runtime)
        pixel_values = pixel_values.to(dtype=runtime["dtype"])
        if runtime["move_pixel_values"]:
            pixel_values = pixel_values.to(runtime["device"])

        generation_config = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "eos_token_id": tokenizer.eos_token_id,
        }

        chat_kwargs = {
            "tokenizer": tokenizer,
            "pixel_values": pixel_values,
            "question": question,
            "generation_config": generation_config,
            "verbose": self.verbose,
        }
        if len(num_patches_list) > 1:
            chat_kwargs["num_patches_list"] = num_patches_list

        with torch.inference_mode():
            return model.chat(**chat_kwargs)

    def _cache_key(self):
        return "|".join(
            [
                self.model_path,
                self.repo_path,
                self.device,
                self.torch_dtype_name,
                str(self.load_in_8bit),
                str(self.load_in_4bit),
            ]
        )

    def _load_runtime(self):
        cache_key = self._cache_key()
        if cache_key in self._runtime_cache:
            return self._runtime_cache[cache_key]

        self._ensure_dependencies()

        import torch
        import torchvision.transforms as T
        from PIL import Image
        from torchvision.transforms.functional import InterpolationMode
        repo_root = Path(self.repo_path)
        if not repo_root.exists():
            raise RuntimeError(f"InternVL repo path does not exist: {self.repo_path}")
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from internvl.model import load_model_and_tokenizer

        loader_args = SimpleNamespace(
            checkpoint=self.model_path,
            auto=(self.device == "auto" and torch.cuda.device_count() > 1),
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit,
        )
        model, tokenizer = load_model_and_tokenizer(loader_args)

        dtype = self._resolve_torch_dtype(torch)
        resolved_device = self._resolve_device(torch)
        move_pixel_values = not loader_args.auto and not self.load_in_8bit and not self.load_in_4bit

        image_size = int(
            getattr(model.config, "force_image_size", None)
            or getattr(model.config.vision_config, "image_size", self.frame_image_size)
            or self.frame_image_size
        )

        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        runtime = {
            "torch": torch,
            "Image": Image,
            "model": model,
            "tokenizer": tokenizer,
            "dtype": dtype,
            "device": resolved_device,
            "image_size": image_size,
            "transform": transform,
            "move_pixel_values": move_pixel_values,
        }
        self._runtime_cache[cache_key] = runtime
        return runtime

    def _ensure_dependencies(self):
        missing = []
        for module_name in ("transformers", "PIL", "torchvision"):
            probe = "PIL" if module_name == "PIL" else module_name
            if importlib.util.find_spec(probe) is None:
                missing.append(module_name)
        if missing:
            deps = ", ".join(missing)
            raise RuntimeError(
                "InternVL provider requires additional dependencies. "
                f"Missing modules: {deps}. "
                "Install the updated requirements in the slowfast environment before running."
            )

    def _resolve_torch_dtype(self, torch):
        if self.torch_dtype_name == "float16":
            return torch.float16
        if self.torch_dtype_name == "float32":
            return torch.float32
        if self.torch_dtype_name == "bfloat16" and torch.cuda.is_available():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32

    def _resolve_device(self, torch):
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _sample_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if not frames:
            raise ValueError("cannot run VLM on an empty clip")
        count = min(self.sampled_frames, len(frames))
        if count <= 1:
            return [frames[0]]

        last_index = len(frames) - 1
        anchor_indices = [0, last_index // 2, last_index]
        if count <= 3:
            chosen = sorted(set(anchor_indices[:count]))
        else:
            extra_needed = count - len(set(anchor_indices))
            extra_indices = np.linspace(0, last_index, num=extra_needed + 2)[1:-1]
            chosen = sorted(
                set(anchor_indices + [int(round(index)) for index in extra_indices])
            )
            if len(chosen) > count:
                chosen = chosen[:count]
            while len(chosen) < count:
                for candidate in range(last_index + 1):
                    if candidate not in chosen:
                        chosen.append(candidate)
                    if len(chosen) == count:
                        break
                chosen = sorted(chosen)
        return [frames[index] for index in chosen]

    def _build_question(self, prompt, num_frames):
        frame_lines = [f"Frame-{index + 1}: <image>" for index in range(num_frames)]
        return "\n".join(frame_lines + [prompt])

    def _build_pixel_values(self, sampled_frames, runtime):
        torch = runtime["torch"]
        Image = runtime["Image"]
        transform = runtime["transform"]

        tensors = []
        num_patches_list = []
        for frame in sampled_frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            image_tensor = transform(pil_image).unsqueeze(0)
            tensors.append(image_tensor)
            num_patches_list.append(1)

        pixel_values = torch.cat(tensors, dim=0)
        return pixel_values, num_patches_list


def _sample_frames(frames: List[np.ndarray], count: int) -> List[np.ndarray]:
    if not frames:
        raise ValueError("cannot run VLM on an empty clip")
    count = min(count, len(frames))
    if count <= 1:
        return [frames[0]]
    last_index = len(frames) - 1
    anchor_indices = [0, last_index // 2, last_index]
    if count <= 3:
        chosen = sorted(set(anchor_indices[:count]))
    else:
        extra_needed = count - len(set(anchor_indices))
        extra_indices = np.linspace(0, last_index, num=extra_needed + 2)[1:-1]
        chosen = sorted(set(anchor_indices + [int(round(i)) for i in extra_indices]))
        if len(chosen) > count:
            chosen = chosen[:count]
        while len(chosen) < count:
            for candidate in range(last_index + 1):
                if candidate not in chosen:
                    chosen.append(candidate)
                if len(chosen) == count:
                    break
            chosen = sorted(chosen)
    return [frames[i] for i in chosen]


class BedrockVLMProvider:
    def __init__(self, config):
        self.model_id = str(config.get("model_id", "us.anthropic.claude-haiku-4-5-20251001-v1:0"))
        self.region = str(config.get("region", "us-east-1"))
        self.max_tokens = int(config.get("max_tokens", 256))
        self.sampled_frames = int(config.get("sampled_frames", 6))
        self._client = None

    def _get_client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def invoke(self, clip_record, prompt):
        sampled = _sample_frames(clip_record["frames"], self.sampled_frames)

        content = []
        for frame in sampled:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            success, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                continue
            content.append({
                "image": {
                    "format": "jpeg",
                    "source": {"bytes": buf.tobytes()},
                }
            })
        content.append({"text": prompt})

        response = self._get_client().converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": content}],
            inferenceConfig={"maxTokens": self.max_tokens, "temperature": 0.0},
        )
        return response["output"]["message"]["content"][0]["text"]


class VLMRefiner:
    def __init__(self, config):
        self.provider_name = str(config.get("provider", "mock")).lower()
        self.sampled_frames = int(config.get("sampled_frames", 6))
        self.prompt_style = config.get("prompt_style", "binary_fight")

        if self.provider_name == "mock":
            self.provider = MockVLMProvider()
        elif self.provider_name == "internvl":
            self.provider = InternVLVLMProvider(config)
        elif self.provider_name == "bedrock":
            self.provider = BedrockVLMProvider(config)
        else:
            raise ValueError(f"unsupported VLM provider: {self.provider_name}")

    def score_event(self, event_frames, event_meta=None):
        """Score an entire event window as a single VLM call (event-level VLM)."""
        meta = event_meta or {}
        duration_sec = float(meta.get("duration_sec", 0.0))
        prompt = build_event_fight_prompt(num_frames=len(event_frames), duration_sec=duration_sec)
        event_record = {
            "frames": event_frames,
            "fighting_prob": float(meta.get("peak_score", 0.5)),
            "uncertainty": 0.5,
            "motion_summary": f"event duration {duration_sec:.1f}s",
        }
        raw = self.provider.invoke(event_record, prompt)
        parsed = parse_vlm_response(raw)
        label = parsed.get("label", "non_fight")
        confidence = float(parsed.get("confidence", 0.5))
        score = confidence if label == "fight" else 1.0 - confidence
        return {
            "prompt": prompt,
            "raw_response": raw,
            "parsed": parsed,
            "score": max(0.0, min(1.0, score)),
        }

    def score_clip(self, clip_record):
        prompt = build_binary_fight_prompt(
            clip_record["motion_summary"],
            num_frames=min(self.sampled_frames, len(clip_record.get("frames", []))),
        )
        raw = self.provider.invoke(clip_record, prompt)
        parsed = parse_vlm_response(raw)
        label = parsed.get("label", "non_fight")
        confidence = float(parsed.get("confidence", 0.5))
        score = confidence if label == "fight" else 1.0 - confidence
        return {
            "prompt": prompt,
            "raw_response": raw,
            "parsed": parsed,
            "score": max(0.0, min(1.0, score)),
        }

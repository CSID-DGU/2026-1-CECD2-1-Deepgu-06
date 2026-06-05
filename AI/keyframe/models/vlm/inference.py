import cv2
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

from models.vlm.parser import parse_vlm_output, normalize_vlm_output
from models.vlm.prompts import load_prompt
from utils.device import choose_torch_device


class InternVL:
    """InternVL2 계열 (OpenGVLab/InternVL2-*).

    model.chat() API 사용. 멀티프레임은 질문에 <image> 플레이스홀더를
    프레임 수만큼 삽입하고 num_patches_list=[1,...,1] 로 전달.
    """

    def __init__(self, model_name="OpenGVLab/InternVL2-8B", device=None,
                 preferred_gpu_indices=None, prompt_path=None,
                 max_new_tokens=128):
        print(f" InternVL 로딩: {model_name}")
        self.device = device or choose_torch_device(
            preferred_gpu_indices=preferred_gpu_indices, allow_cpu_fallback=True
        )
        self.max_new_tokens = max_new_tokens
        self.prompt = load_prompt(prompt_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map={"": self.device},
        ).eval()

        # 모델 설정에서 image_size 읽기, 없으면 448
        cfg = getattr(self.model, "config", None)
        image_size = (
            getattr(cfg, "force_image_size", None)
            or getattr(getattr(cfg, "vision_config", None), "image_size", None)
            or 448
        )
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        print(" InternVL 로드 완료")

    def predict(self, frames):
        """frames: List[np.ndarray(BGR)] → dict(label, confidence, evidence)"""
        pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        pixel_values = torch.stack([self.transform(img) for img in pil_images])
        pixel_values = pixel_values.to(self.device).to(torch.bfloat16)
        num_patches_list = [1] * len(frames)

        frame_lines = [f"Frame-{i + 1}: <image>" for i in range(len(frames))]
        question = "\n".join(frame_lines + [self.prompt])

        gen_config = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "temperature": 0.0,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        with torch.inference_mode():
            response = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=gen_config,
                num_patches_list=num_patches_list,
            )

        return parse_vlm_output(response)


class Qwen2VL:
    """Qwen2-VL 계열 (Qwen/Qwen2-VL-*).

    generate() API 사용. 이미지를 메시지 content 리스트로 전달하며
    프레임 수 제한 없고 native resolution 지원.
    """

    def __init__(self, model_name="Qwen/Qwen2-VL-7B-Instruct", device=None,
                 preferred_gpu_indices=None, prompt_path=None,
                 max_new_tokens=128, min_pixels=256 * 28 * 28,
                 max_pixels=1280 * 28 * 28):
        print(f" Qwen2-VL 로딩: {model_name}")
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        self.device = device or choose_torch_device(
            preferred_gpu_indices=preferred_gpu_indices, allow_cpu_fallback=True
        )
        self.max_new_tokens = max_new_tokens
        self.prompt = load_prompt(prompt_path)

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
        ).eval()

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        # qwen_vl_utils 선택적 임포트
        try:
            from qwen_vl_utils import process_vision_info as _pvi
            self._process_vision_info = _pvi
        except ImportError:
            self._process_vision_info = None

        print(" Qwen2-VL 로드 완료")

    def predict(self, frames):
        """frames: List[np.ndarray(BGR)] → dict(label, confidence, evidence)"""
        pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]

        # 저해상도 CCTV(e.g. 320×240) 대응: 짧은 변이 448px 미만이면 업스케일
        resized = []
        for img in pil_images:
            w, h = img.size
            if min(w, h) < 448:
                scale = 448 / min(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            resized.append(img)
        pil_images = resized

        messages = [{
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in pil_images],
                {"type": "text", "text": self.prompt},
            ],
        }]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if self._process_vision_info is not None:
            image_inputs, _ = self._process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, padding=True, return_tensors="pt"
            )
        else:
            inputs = self.processor(
                text=[text], images=pil_images, padding=True, return_tensors="pt"
            )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )

        trimmed = [
            output_ids[i][len(inputs["input_ids"][i]):]
            for i in range(len(output_ids))
        ]
        response = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        return parse_vlm_output(response)


class Phi35Vision:
    """Phi-3.5-Vision-Instruct (microsoft/Phi-3.5-vision-instruct).

    AutoModelForCausalLM + AutoProcessor. 멀티이미지는 <|image_N|> 태그로 삽입.
    """

    def __init__(self, model_name="microsoft/Phi-3.5-vision-instruct", device=None,
                 preferred_gpu_indices=None, prompt_path=None, max_new_tokens=128):
        from transformers import AutoModelForCausalLM, AutoProcessor

        print(f" Phi-3.5-Vision 로딩: {model_name}")
        self.device = device or choose_torch_device(
            preferred_gpu_indices=preferred_gpu_indices, allow_cpu_fallback=True
        )
        self.max_new_tokens = max_new_tokens
        self.prompt = load_prompt(prompt_path)

        device_map = {"": self.device}
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device_map,
            _attn_implementation="eager",
        ).eval()

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            num_crops=4,
        )
        print(" Phi-3.5-Vision 로드 완료")

    def predict(self, frames):
        """frames: List[np.ndarray(BGR)] → dict(label, confidence, evidence)"""
        pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]

        image_tags = "".join(f"<|image_{i+1}|>\n" for i in range(len(pil_images)))
        user_content = image_tags + self.prompt
        messages = [{"role": "user", "content": user_content}]

        prompt_text = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(prompt_text, pil_images, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        trimmed = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.processor.decode(trimmed, skip_special_tokens=True)
        return parse_vlm_output(response)


def load_vlm(model_name, device=None, prompt_path=None, **kwargs):
    """model_name으로 VLM 클래스를 자동 선택해 반환."""
    name_lower = model_name.lower()
    if "qwen" in name_lower:
        return Qwen2VL(model_name=model_name, device=device, prompt_path=prompt_path, **kwargs)
    if "phi" in name_lower:
        return Phi35Vision(model_name=model_name, device=device, prompt_path=prompt_path, **kwargs)
    return InternVL(model_name=model_name, device=device, prompt_path=prompt_path, **kwargs)

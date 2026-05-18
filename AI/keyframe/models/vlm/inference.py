import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from torchvision import transforms
from models.vlm.parser import parse_vlm_output, normalize_vlm_output
from models.vlm.prompts import load_prompt
from utils.device import choose_torch_device



# VLM 클래스
class InternVL:
    def __init__(self, model_name="OpenGVLab/InternVL2-8B", device=None, preferred_gpu_indices=None, prompt_path=None):

        print(" InternVL 모델 로딩 중...")

        self.device = device or choose_torch_device(
            preferred_gpu_indices=preferred_gpu_indices,
            allow_cpu_fallback=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map={"": self.device}
        ).eval()

        self.prompt = load_prompt(prompt_path)

        print(" 모델 로딩 완료")


    def preprocess_frames(self, frames):
        
        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        images = []

        for frame in frames:
            img = Image.fromarray(frame)
            img = transform(img)
            images.append(img)

        return images


    def predict(self, frames):

        images = self.preprocess_frames(frames)
        images = torch.stack(images).to(self.device).to(torch.bfloat16)
        response = self.model.chat(
            tokenizer=self.tokenizer,
            pixel_values=images,
            question=self.prompt,
            generation_config={
                "max_new_tokens": 100,
                "do_sample": False,
                "temperature": 0.0
            }
        )

        parsed = parse_vlm_output(response)
        parsed = normalize_vlm_output(parsed)

        return parsed

import torch
from PIL import Image
import numpy as np
from transformers import AutoModel, AutoTokenizer
from torchvision import transforms


class InternVL:
    def __init__(self, model_name="OpenGVLab/InternVL2-8B"):

        print(" InternVL 모델 로딩 중...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map={"": "cuda:1"}
        ).eval()

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
            img = Image.fromarray(frame)  # numpy → PIL
            img = transform(img)          # PIL → Tensor
            images.append(img)

        return images

    def build_prompt(self):
        return (
            "You are an expert in video anomaly detection.\n"
            "Analyze the following frames and answer:\n"
            "1. Is there any abnormal behavior?\n"
            "2. If yes, describe it briefly.\n"
            "Answer in JSON format:\n"
            '{"label": "anomaly or normal", "description": "..."}'
        )

    def predict(self, frames):

        images = self.preprocess_frames(frames)
        images = torch.stack(images).to("cuda:1").to(torch.bfloat16)

        """
        vlm 들어가기 직전 이미지 텐서 상태 확인
        
        print(type(images[0]))
        print(images.shape, images.dtype, images.device)
        """

        prompt = self.build_prompt()

        generation_config = dict(
            max_new_tokens=100,
            do_sample=False,
            temperature=0.0
        )

        response = self.model.chat(
            tokenizer=self.tokenizer,
            pixel_values=images,
            question=prompt,
            generation_config=generation_config
        )

        return response
    
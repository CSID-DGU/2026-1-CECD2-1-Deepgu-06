import torch
from ultralytics import YOLO

print("GPU 사용 가능:", torch.cuda.is_available())
print("GPU 이름:", torch.cuda.get_device_name(0))

model = YOLO("yolov8n.pt")
print("YOLO 모델 로드 성공")
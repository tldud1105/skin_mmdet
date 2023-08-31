import torch
from torchvision.models import resnet18
from thop import profile

from mmdet.apis import inference_detector, init_detector

config_file = 'rtmdet_tiny_8xb32-300e_coco.py'
checkpoint_file = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')  # device='cpu' or device='cuda:0'

input_data = torch.randn(1, 3, 224, 224)  # Example input data

flops, params = profile(model, inputs=(input_data,))
print(f"FLOPs: {flops / 1e9} G")

# 모델의 매개 변수 수 계산
num_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {num_params}")
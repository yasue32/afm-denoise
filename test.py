import torch
from PIL import Image
import numpy as np
from realesrgan import RealESRGAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4plus.pth')

path_to_image = 'inputs/lr_lion.png'
image = Image.open(path_to_image).convert('RGB')

sr_image = model.predict(image)

sr_image.save('results/sr_lion.png')
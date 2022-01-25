from torchvision.transforms import functional as tvF
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch
import cv2


class DegradationScore:
    def __init__(self, ch=3):
        self.blurScore = BlurScore()
        self.noiseScore = NoiseScore(ch=ch)

    def __call__(self, imgs):
        return *self.blurScore(imgs), *self.noiseScore(imgs)


class NoiseScore():
    def __init__(self, ch):
        kernel = torch.Tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]])
        self.kernel1 = kernel.reshape(1, 1, *kernel.shape).repeat(ch, ch, 1, 1) # [output_ch, input_ch, kernelH, kernelW]
        self.kernel2 = torch.transpose(self.kernel1, 2, 3)

    def __call__(self, imgs):
        scores = {}
        imgs = self.array2Tensor(imgs)
        scores = self.calc(imgs)
        scores = {key: val for key, val in list(enumerate(scores))}
        scores_sorted = np.array(sorted(scores.items(), key=lambda x:x[1], reverse=True))
        return scores_sorted[:, 0], scores_sorted[:, 1] # idxs, scores

    def calc(self, imgs):
        score_map = (F.conv2d(imgs, self.kernel1) - F.conv2d(imgs, self.kernel2)).abs()
        scores = score_map.mean(dim=(1, 2, 3)) / 3 # [B, C, H, W] -> [B], * Divide 3 is an operation to match the score when applied to a single image read by L type.
        return scores.to('cpu').detach().numpy()

    def array2Tensor(self, imgs):
        return torch.from_numpy(imgs).permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]


class BlurScore():
    def __init__(self):
        pass

    def __call__(self, imgs):
        scores = {}
        for idx, img in enumerate(imgs):
            scores[idx] = self.calc(img)
    
        scores_sorted = np.array(sorted(scores.items(), key=lambda x:x[1], reverse=True))
        return scores_sorted[:, 0], scores_sorted[:, 1] # idxs, scores

    def calc(self, img):
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        # print(type(img), img, img.shape)
        return cv2.Laplacian((img * 255).astype(np.uint8), cv2.CV_64F).var()
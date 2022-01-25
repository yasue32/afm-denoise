import os
os.environ["CUDA_VISIBLE_DEVICES"] = "9"

import cv2
import numpy as np
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sift_flow_torch import SiftFlowTorch
from third_party.flowiz import flowiz


def find_local_matches(desc1, desc2, kernel_size=7):
    # Computes the correlation between each pixel on desc1 with all neighbors
    # inside a window of size (kernel_size, kernel_size) on desc2. The match
    # vector if then computed by linking each pixel on desc1 with
    # the pixel with desc2 with the highest correlation.
    #
    # This approch requires a lot of memory to build the unfolded descriptor.
    # A better approach is to use the Correlation package from e.g.
    # https://github.com/NVIDIA/flownet2-pytorch/tree/master/networks/correlation_package
    desc2_unfolded = F.unfold(desc2, kernel_size, padding=kernel_size//2)
    desc2_unfolded = desc2_unfolded.reshape(
        1, desc2.shape[1], kernel_size*kernel_size, desc2.shape[2], desc2.shape[3])
    desc1 = desc1.unsqueeze(dim=2)
    correlation = torch.sum(desc1 * desc2_unfolded, dim=1)
    _, match_idx = torch.max(correlation, dim=1)
    hmatch = torch.fmod(match_idx, kernel_size) - kernel_size // 2
    vmatch = match_idx // kernel_size - kernel_size // 2
    matches = torch.cat((hmatch, vmatch), dim=0)
    return matches


sift_step_size = 2
image_resize_factor = 1

sift_flow = SiftFlowTorch(
    cell_size=1,
    step_size=sift_step_size,
    is_boundary_included=True,
    num_bins=8,
    cuda=True,
    fp16=True,
    return_numpy=False)
imgs = [
    cv2.imread('orig_img/20211115/00004.png'),
    cv2.imread('orig_img/20211115/00005.png')
]
imgs = [cv2.resize(im, (im.shape[1]//image_resize_factor, im.shape[0]//image_resize_factor)) for im in imgs]
# print(imgs)
descs = sift_flow.extract_descriptor(imgs)

flow = find_local_matches(descs[0:1], descs[1:2], 7)

# Show input images
fig=plt.figure(figsize=(10, 10))
fig.add_subplot(2, 2, 1)
plt.title("image1")
plt.imshow(cv2.cvtColor(imgs[0], cv2.COLOR_BGR2RGB))
fig.add_subplot(2, 2, 2)
plt.title("image2")
plt.imshow(cv2.cvtColor(imgs[1], cv2.COLOR_BGR2RGB))
flow = flow.permute(1, 2, 0).detach().cpu().numpy()
flow_img = flowiz.convert_from_flow(flow)
fig.add_subplot(2, 2, 3)
plt.title("image1 -> image2")
plt.imshow(flow_img)
fig.savefig("flow_img4.png")

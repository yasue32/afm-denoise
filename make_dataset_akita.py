import os
import glob
import shutil

import torch
import torch.nn.functional as F
from sift_flow_torch import SiftFlowTorch
from third_party.flowiz import flowiz
import cv2
from tqdm import tqdm
import alignment
import numpy as np
from copy import deepcopy

device=torch.device('cuda')

#DATA_DIR = 'ext_clean_img'
DATA_DIR = "test_imgs"
SAVE_DIR = 'test_dataset_per_sequence_aligned2'

save_sift_flow = False
align = True

sift_flow = SiftFlowTorch(
    cell_size=1,
    step_size=1,
    is_boundary_included=False,
    num_bins=8,
    cuda=True,
    fp16=True,
    return_numpy=False)

### find match
def find_local_matches(desc1, desc2, kernel_size=5):
    """SIFT特徴量の一致を検索する
    desc1: sift_flow, 1枚目
    desc2: sift_flow, 2枚目
    kernel_size: int, flowのノルムの最大値
    """ 
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

def cal_noise_level(flows):
    #flow[10,10,2,128,128]を受け取って、ノイズレベルを返す
    # [[noise0, index0],[noise1, index1],...] 
    n_burst = len(flows)
    l1 = len(flows[0,0])
    l2 = len(flows[0,0,0])
    flows = flows.permute(0,1,3,4,2)
    noise_level_list = []

    for j in range(n_burst):
        noise_level = 0
        for k in range(n_burst):
            if j == k:
                continue
            vector_map = flows[j,k]  # flow img-j to img-k
            vector_map = vector_map.permute(2,0,1)
            #print(vector_map.shape)
            tmp = torch.var(vector_map[0], dim=1) + torch.var(vector_map[1], dim=1)
            noise_level += torch.sum(tmp).item()
        noise_level_list.append([noise_level, j])
    noise_level_list = sorted(noise_level_list)

    return noise_level_list

### flow 計算
# optical flow を計算する flows.shape = [10,10,2,128,128]
# flowが画像サイズより小さくなる (img128px -> flow 120pxなど) 周りはゼロで埋める
def cal_flow(model, imgs, kernel_size=8):
    l = len(imgs)

    il1, il2, il3 = imgs[0].shape
    #print(imgs, l, imgs[0].shape)
    descs = model.extract_descriptor(imgs)
    # print('Descriptor shape:', descs.shape)
    flow_test = find_local_matches(descs[0:1], descs[1:2])
    #print(flow_test.shape)
    l0, l1, l2 = flow_test.shape
    diff1 = il1 - l1
    diff2 = il2 - l2
    flows = torch.zeros(l, l, l0, l1 + diff1, l2 + diff2).to(device)
    for img1 in range(l):
        for img2 in range(l):
            matches = find_local_matches(descs[img1:img1+1], descs[img2:img2+1], kernel_size=kernel_size)
            flows[img1, img2, :, int(diff1/2):int(l1+diff1/2), int(diff2/2):int(l2+diff2/2)] = matches 
    return flows


i = 0
same_count = 0

for pathes in tqdm(zip(*[iter(sorted(glob.glob(os.path.join(DATA_DIR, '**/*.png'))))]*10)):
    assert pathes[0][-5] == '0'
    save_dir = os.path.join(SAVE_DIR, str(i).zfill(4))
    os.makedirs(save_dir, exist_ok=True)

    imgs = []
    # print(pathes)
    for index, path in enumerate(pathes[::-1]):
        save_path = os.path.join(save_dir, str(index).zfill(4)+".png")
        if align:
            if index == 0:
                target_img = cv2.imread(path)
                shutil.copy(path, save_path)
            else:
                input_img = cv2.imread(path)
                aligned = alignment.affine_align(target_img, [deepcopy(input_img)])[0]
                # print(aligned - input_img)
                if np.sum(aligned - input_img) == 0:
                    same_count += 1
                cv2.imwrite(save_path, aligned)
        else:
            shutil.copy(path, save_path)
        
        imgs.append(cv2.imread(save_path))

    ## ptを保存したいときは外す
    if save_sift_flow:  
        flows = cal_flow(sift_flow, imgs, kernel_size=15)
        for index, path in enumerate(pathes):
            save_path = os.path.join(save_dir, str(index).zfill(4)+".pt")
            torch.save(flows[index], save_path)

    if align:
        print(same_count)
    i += 1

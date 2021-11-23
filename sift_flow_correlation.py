# coding: UTF-8
### GPU 指定
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"

### import 
# correlation より先にtorchをimport
import torch
#import correlation_cuda
device=torch.device('cuda')

import cv2
import numpy as np
import time

import torch.nn.functional as F
import matplotlib.pyplot as plt
import statistics

from sift_flow_torch import SiftFlowTorch
from third_party.flowiz import flowiz
import glob

#from correlation_package.correlation import Correlation



### find match
def find_local_matches(desc1, desc2, kernel_size=9):
    """SIFT特徴量の一致を検索する
    desc1: sift_flow, １枚目
    desc2: sift_flow, ２枚目
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

# def find_local_matches2(desc1, desc2, kernel_size=9):
#     corr = Correlation(pad_size=20, kernel_size=kernel_size, 
#     max_displacement=20, stride1=1, stride2=2, corr_multiply=1)
#     matches = corr(desc1, desc2)
#     return matches
# flow2 = find_local_matches2(descs[0:1], descs[1:2], 9)
# print(flow2.shape)
# print(flow2[0])


# def add_noise(inputs,noise_factor=0.3):
#     x = int(np.random.rand()*inputs.shape[0])
#     inputs[x] = 0
#     return inputs


def read_imgs(load_filepath, load_filename, index,image_resize_factor = 1):
    """画像をまとめて読み込む関数
    filepath: str, 画像ファイルのpath
    index: list, 画像のindex
    image_resize_factor: int, 画像をリサイズする場合
    """
    imgs = [cv2.imread(load_filepath+"/"+load_filename.format(i)) for i in index]
    #print([load_filepath+"/"+load_filename.format(i) for i in index])
    imgs = [cv2.resize(im, (im.shape[1]//image_resize_factor, im.shape[0]//image_resize_factor)) for im in imgs]
    print("img size: ", imgs[0].shape)
    #imgs = [add_noise(im) for im in imgs]
    return imgs


### n枚のimgsを受け取り、まとめてflowを計算して返す
def cal_flow(model, imgs, kernel_size=7):
    l = len(imgs)
    descs = model.extract_descriptor(imgs)
    print('Descriptor shape:', descs.shape)
    flow_test = find_local_matches(descs[0:1], descs[1:2], kernel_size)
    #flow=[[]*l for i in range(l)]
    flow_list = []
    flow = torch.empty(l, *flow_test.shape).to(device)
    for img1 in range(l):
        for img2 in range(l):
            print(img1, img2)
            flow[img2] = find_local_matches(descs[img1:img1+1], descs[img2:img2+1], kernel_size)
        flow_list.append(flow)
    #print(flow.device, flow.shape)
    return flow


### 工事中
def visualize_flow(flow, filename="flow_img", den=1/10):
    # Show optical flow
    flow = flow.permute(1, 2, 0).detach().cpu().numpy()
    flow_img = flowiz.convert_from_flow(flow)
    flow_u = flow[::int(-1//den),::int(1//den),0]
    flow_v = flow[::int(-1//den),::int(1//den),1]*-1
    y = []
    x = []
    for k in range(len(flow_u)):
        x.append(statistics.pvariance(map(float, flow_u[:][k]))+statistics.pvariance(map(float, flow_v[:][k])))
        y.append(k)
    
    fig = plt.figure(figsize = (20,10))
    fig.add_subplot(1, 2, 1)
    plt.imshow(flow_img)
    fig.add_subplot(1, 2, 2)
    plt.scatter(y,x,s=3)
    plt.savefig(filename)
    plt.show()


### flow 計算
def flow_main(load_filepath, load_filename, save_filepath, n_burst=10, n_set=9, 
sift_step_size = 1, kernel_size = 8):
    sift_flow = SiftFlowTorch(
        cell_size=2,
        step_size=sift_step_size,
        is_boundary_included=False,
        num_bins=8,
        cuda=True,
        fp16=True,
        return_numpy=False)
    
    os.makedirs(save_filepath, exist_ok=True)
    files = glob.glob(load_filepath + "/*" + load_filename.split(".")[-1])
    n_set = int(len(files)/n_burst)
    flow = []
    for i in range(n_set):
        print("=="*10)
        t1 = time.time()
        index = list(range(i*n_burst, (i+1)*n_burst))
        imgs = read_imgs(load_filepath,load_filename, index)
        flow = cal_flow(sift_flow, imgs, kernel_size=kernel_size)
        
        for j in range(n_burst):
            filename = "{:05}".format(i*n_burst + j)+".pt"
        # 画像ごとにtorchのflowを保存する
            torch.save(flow, save_filepath+"/"+filename)
            t2 = time.time()
        print("img set{}: {:.2f}s".format(i, t2-t1), ", flow_size", flow[0].shape)

### flow 計算の実行
#flow_main("TIFF files", "5um CNT.0_{:05}_1.spm.tif","result_flow", n_set=9)
#sub = "211022"
#flow_main("orig_img/"+sub, "{:05}.png","result_flow/"+sub, kernel_size=5)







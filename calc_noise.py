# coding: UTF-8
### GPU 指定
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"


### import 
import torch
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


# 各行のflowの分散をノイズレベルとする
def cal_noise_level(load_filepath, save_filepath, orig_img_filepath, n_burst=10, n_set=9, v=True):
    files = glob.glob(load_filepath + "/*")
    n_set = int(len(files)/n_burst)
    for i in range(n_set):
        flow = []
        for j in range(n_burst):
            filename = "{:05}".format(i*n_burst + j)+".pt"
            flow.append(torch.load(load_filepath+"/"+filename))
            flow[-1] = flow[-1].permute(0,2,3,1)
        print(flow[0].shape)
        l1 = len(flow[0][0])
        l2 = len(flow[0][0,0])

        os.makedirs(save_filepath, exist_ok=True)
        for j in range(n_burst):
            norm_map = torch.zeros(l1,l2).to(device)
            norm_map_tmp = torch.zeros(l1,l2).to(device)
            for k in range(n_burst):
                if j==k:
                    continue
                vector_map = flow[j][k]  # flow img-j to img-k
                vector_map2 = vector_map.permute(2,0,1)
                # vector_map2 = vector_map
                print(vector_map2.shape)
                #tmp = torch.norm(vector_map, dim=2) # flowベクトルをnormに
                tmp = torch.var(vector_map2[0], dim=1) + torch.var(vector_map2[1], dim=1)
                #norm_map += tmp * tmp  # 要素ごとの積
                print(tmp.shape)
                norm_map_tmp[:] = tmp                
                norm_map += norm_map_tmp.T
                # for u in range(l1):
                #     for v in range(l2):
                #         norm_map[u,v] += torch.norm(flow[j,k,u,v]) # flow[u][v]のnorm
            
            norm_map -= torch.min(norm_map) 
            #print(norm_map)
            print(torch.max(norm_map))
            filename = "{:05}".format(i*n_burst+j)
            orig_img = cv2.imread(orig_img_filepath+"/"+filename+".png")
            torch.save(norm_map, save_filepath+"/"+filename+".pt")
            if v:
                visualize_noise(norm_map, save_filepath, filename, orig_img)    
  
            

# 1に加えて、上下ｎ行の平均をノイズレベルとする(工事中)
def cal_noise_level2(load_filepath, save_filepath, n_burst=10, n_set=9):
    for i in range(n_set):
        filename = "{:05}-{:05}".format(i*n_burst, (i+1)*n_burst-1)+".pt"
        flow = torch.load(load_filepath+"/"+filename)
        flow = flow.permute(0,1,3,4,2)
        print(flow.shape)
        l1 = len(flow[0,0])
        l2 = len(flow[0,0,0])

        os.makedirs(save_filepath, exist_ok=True)
        for j in range(n_burst):
            norm_map = torch.zeros(l1,l2).to(device)
            for k in range(n_burst):
                if j==k:
                    continue
                vector_map = flow[j,k]  # flow  img-j to img-k
                vector_map2 = vector_map.permute(2,0,1)
                print(vector_map2.shape)
                #tmp = torch.norm(vector_map, dim=2) # flowベクトルをnormに
                tmp = torch.var(vector_map2[0], dim=1) + torch.var(vector_map2[1], dim=1)
                #norm_map += tmp * tmp  # 要素ごとの積
                print(tmp.shape)
                norm_map[:] = tmp
                norm_map = norm_map.T
                # for u in range(l1):
                #     for v in range(l2):
                #         norm_map[u,v] += torch.norm(flow[j,k,u,v]) # flow[u][v]のnorm
            
            norm_map -= torch.min(norm_map) 
            #norm_map = 
            #print(norm_map)
            print(torch.max(norm_map))
            filename = "{:05}".format(i*n_burst+j)
            orig_img = cv2.imread(orig_img_filepath+"/"+filename)
            torch.save(norm_map, save_filepath+"/"+filename+".pt")
            visualize_noise(norm_map, save_filepath, filename, orig_img)    
            

def visualize_noise(noise_map, save_filepath, filename, orig_img):
    noise_map = noise_map.cpu().numpy()
    img = 255*(noise_map - np.min(noise_map)/np.max(noise_map))
    print(orig_img.shape, "->", img.shape)
    orig_img = np.array( cv2.resize(orig_img, (img.shape[0], img.shape[1] )))
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY) 
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR) 
    #print(orig_img)
    for line, noise in enumerate(noise_map):
        if sum(noise)>1600:
            #orig_img[line] = orig_img[line] *0.7
            orig_img[line,:,1] = orig_img[line,:,1] *0.5
            #for i in range(len(orig_img[line])):
            #    orig_img[line][i] = int(orig_img[line][i])

    #alpha = 0.99
    #img = (img*(1-alpha) + alpha*orig_img)
    plt.figure(figsize = (10,10))
    #plt.imshow(img)
    #plt.imshow(orig_img, cmap = "gray")
    plt.imshow(orig_img)
    plt.savefig(save_filepath+"/"+filename+".png")
    plt.show()
    plt.clf()
    plt.close()


### flow からノイズレベルを推定する
#cal_noise_level("result_flow", "result_noise")


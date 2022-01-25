# coding: UTF-8
### GPU 指定
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "4"


### import 
import torch
device=torch.device('cuda')

import sys
import cv2
import numpy as np
import time

import torch.nn.functional as F
import matplotlib.pyplot as plt
import statistics

from sift_flow_torch import SiftFlowTorch
from third_party.flowiz import flowiz
import glob
from PIL import Image


def read_imgs(load_filepath, load_filename, index,image_resize_factor = 1):
    """画像をまとめて読み込む関数
    filepath: str, 画像ファイルのpath
    index: list, 画像のindex
    image_resize_factor: int, 画像をリサイズする場合
    """
    print(index)
    imgs = [cv2.imread(load_filepath+"/"+load_filename.format(i)) for i in index]
    #print([load_filepath+"/"+load_filename.format(i) for i in index])
    imgs = [cv2.resize(im, (im.shape[1]//image_resize_factor, im.shape[0]//image_resize_factor)) for im in imgs]
    print("img size: ", imgs[0].shape)
    #imgs = torch.tensor(imgs)
    #print(imgs[0])
    #imgs = torch.squeeze(imgs, dim=-1)
    #print(imgs[0])
    return imgs


def choise_gt(noise_batch):  # ノイズマップ10枚
    gt = 0
    small = sum(sum(noise_batch[0]))
    noisy = []
    for i in range(len(noise_batch)):
        s = sum(sum(noise_batch[i]))
        noisy.append([s, i])
        if s < small:
            small = s
            gt = i
    #noisy = list(range(len(noise_batch)))
    print(gt, end=", ")
    del noisy[gt]
    noisy = sorted(noisy)
    return gt, noisy # 最もきれいな画像のindex、その他の画像のindex(0~n-1, n+1~9)


def make_dataset(load_noise_filepath="result_noise",load_img_filepath="TIFF files", 
                load_img_filename="5um CNT.0_{:05}_1.spm.tif", save_filepath="dataset",
                n_batch=4, n_burst=10, n_set=9, gamma_corr=False):
    os.makedirs(save_filepath, exist_ok=True)
    files = glob.glob(load_noise_filepath + "/*.pt")
    n_set = int(len(files)/n_burst)
    if gamma_corr:
        gamma045LUT = [pow(x/255.0, 1.3/2.2)*255 for x in range(256)]
    print(n_set, "set")
    for i in range(n_set):
        print("=="*10)
        index = list(range(i*n_burst, (i+1)*n_burst))
        imgs = read_imgs(load_img_filepath,load_img_filename, index)
        
        noise_map_list = [torch.load(load_noise_filepath
            + "/" + "{:05}".format(i*n_burst + j)+".pt") for j in range(n_burst)]

        # GPUに移動する
        noise_map = torch.zeros(n_burst, *noise_map_list[0].shape).to(device)
        for j in range(n_burst):
            noise_map[j] = noise_map_list[j].to(device)
        
        # もしn_batchで割り切れなかったらエラーにする
        if (len(imgs[0]) % n_batch) or (noise_map.shape[1] % n_batch) :
            print('Error: batch division faild', file=sys.stderr)
            sys.exit(1)
        
        print("GT=", end="")
        noise_batch_size = int(len(noise_map[0]) / n_batch)
        img_batch_size = int(len(imgs[0]) / n_batch)
        #print(noise_batch_size, img_batch_size)
        #noise_map = torch.tensor(noise_map)

        os.makedirs(save_filepath+"/set{:04}".format(i), exist_ok=True)
        noise_batch = torch.zeros(n_burst, noise_batch_size, noise_batch_size).to(device)
        #img_batch = torch.zeros(n_burst, img_batch_size, img_batch_size, 3) # RGB3チャネルを保持している
        img_batch = [0]*n_burst

        for n1 in range(n_batch): # y方向のbatch分割
            nsy,ngy = n1 *noise_batch_size, (n1+1) * noise_batch_size
            isy,igy = n1 *img_batch_size, (n1+1) * img_batch_size 
            
            for n2 in range(n_batch): # x方向のbatch
                save_filepath_batch = save_filepath + "/set{:04}".format(i) + "/batch{:02}".format(n1*n_batch + n2)
                os.makedirs(save_filepath_batch, exist_ok=True)
                # batch部分を切り出し
                nsx,ngx = n2 * noise_batch_size, (n2+1) * noise_batch_size
                isx,igx = n2 *img_batch_size, (n2+1) * img_batch_size
                for n3 in range(n_burst): # 各画像についてn_burst枚数ずつimg, flowがある
                    noise_batch[n3] = noise_map[n3, nsy:ngy, nsx:ngx]
                    #rint(len(img_batch), len(imgs))
                    img_batch[n3] = imgs[n3][isy:igy, isx:igx]

                # batchを保存する
                gt_index, noisy_index = choise_gt(noise_batch)
                #noise_batch = noise_batch.cpu().numpy()
                #img_batch_numpy = img_batch.cpu().numpy()
                pil_img = Image.fromarray(img_batch[gt_index]).convert("L")
                if gamma_corr:
                    pil_img = pil_img.point(gamma045LUT)
                #pil_img.save(save_filepath_batch+"/gt.png")
                pil_img.save(save_filepath_batch+f"/gt{gt_index}.png")

                #cv2.imwrite(save_filepath_batch+"/gt.png", img_batch_numpy[gt_index])
                with open(save_filepath+"/sep_trainlist.txt", mode='a') as f:
                    
                    for _, n  in (noisy_index):
                        pil_img = Image.fromarray(img_batch[n]).convert("L")
                        if gamma_corr:
                            pil_img = pil_img.point(gamma045LUT)
                        filename = "input{:03}.png".format(n)
                        pil_img.save(save_filepath_batch + "/" + filename)
                        
                        # file_listに追記
                        if i or  n1 or n2 or n:
                            f.write("\n")
                        f.write("set{:04}".format(i) + "/batch{:02}".format(n1*n_batch + n2) + "/" + filename)
                    
        
        print()
    

#make_dataset()

# coding: UTF-8
### GPU 指定
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


### import 
import torch
device=torch.device('cuda')

import argparse
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


def read_imgs(load_filepath, load_filename, index,image_resize_factor = 1, gamma_corr=False, gray=False):
    """画像をまとめて読み込む関数
    filepath: str, 画像ファイルのpath
    index: list, 画像のindex
    image_resize_factor: int, 画像をリサイズする場合
    """
    if gamma_corr:
        gamma045LUT = [pow(x/255.0, 1.3/2.2)*255 for x in range(256)]
    if gray: 
        pil_img = pil_img.convert("L")
    if gamma_corr:
        pil_img = pil_img.point(gamma045LUT)

    print(index)
    #imgs = [cv2.imread(load_filepath+"/"+load_filename.format(i)) for i in index]
    imgs = [cv2.imread(i) for i in index]
    #print([load_filepath+"/"+load_filename.format(i) for i in index])
    imgs = [cv2.resize(im, (im.shape[1]//image_resize_factor, im.shape[0]//image_resize_factor)) for im in imgs]
    print("img size: ", imgs[0].shape)
    
    return imgs


def make_patch(imgs, n_patch=2):
    n_burst = len(imgs)
    # もしn_patchで割り切れなかったらエラーにする
    if (len(imgs[0]) % n_patch):
        print('Error: patch division faild', file=sys.stderr)
        sys.exit(1)
    img_patch_size = int(len(imgs[0][0]) / n_patch)
  
    patches = []

    for n1 in range(n_patch): # y方向のpatch分割
        isy,igy = n1 * img_patch_size, (n1+1) * img_patch_size 
        for n2 in range(n_patch): # x方向のpatch
            isx,igx = n2 * img_patch_size, (n2+1) * img_patch_size
            img_patch = [0]*n_burst
            for n3 in range(n_burst): # n_burst枚数ずつimgがある
                img_patch[n3] = imgs[n3][isy:igy, isx:igx]
            patches.append(img_patch)
    print(len(patches))
    return patches


### find match
def find_local_matches(desc1, desc2, kernel_size=5):
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

### flow 計算
# optical flow を計算する flows.shape = [10,10,2,128,128]
# flowが画像サイズより小さくなる (img128px -> flow 120pxなど) 周りはゼロで埋める
def cal_flow(model, imgs, kernel_size=8):
    l = len(imgs)

    il1, il2, il3 = imgs[0].shape
    #print(imgs, l, imgs[0].shape)
    descs = model.extract_descriptor(imgs)
    print('Descriptor shape:', descs.shape)
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


def save_img_and_flow(p, imgs, noise_level_list, flows, save_filepath, visualize=False):
    save_filepath = save_filepath + "/patch{:02}".format(p)
    os.makedirs(save_filepath, exist_ok=True)
    n_burst = len(imgs)
    file_names = []
    flows = flows.permute(0,1,3,4,2)

    #flowの順番をnoise_levelに合わせて入れ替える
    flows_tmp = torch.empty(flows.shape)
    for i, n in enumerate(noise_level_list):
        flows_tmp[:,i] = flows[n[1]]
    flows = flows_tmp.detach()

    # gt
    n = noise_level_list[0]
    pil_img = Image.fromarray(imgs[n[1]]).convert("L")
    file_name = "gt.png"
    pil_img.save(save_filepath + "/" + file_name)
    flow = flows[n[1]]
    torch.save(flow, save_filepath+"/"+"gt.pt")
    del noise_level_list[0]

    # input
    filepath_index = "/".join(save_filepath.split("/")[2:])
    # ノイズが少ない順にindexをふって保存する
    for i, n in enumerate(noise_level_list):
        pil_img = Image.fromarray(imgs[n[1]]).convert("L")
        file_name = "input{:03}.png".format(i)
        pil_img.save(save_filepath + "/" + file_name)
        file_names.append(filepath_index + "/" + file_name)
        flow = flows[n[1]]
        torch.save(flow, save_filepath+"/"+"input{:03}.pt".format(i))

        if visualize:
            flow_norm = torch.norm(flows[n[1]], dim=-1)
            orig_img = imgs[n[1]]
            noise_map = torch.sum(flow_norm, dim=0).cpu().numpy()
            img = 255*(noise_map - np.min(noise_map)/np.max(noise_map))
            print(orig_img.shape, "->", img.shape)

            orig_img = np.array( cv2.resize(orig_img, (img.shape[0], img.shape[1] )))
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY) 
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR) 
            for line, noise in enumerate(noise_map):
                if sum(noise)>1600: #th
                    orig_img[line,:,1] = orig_img[line,:,1] *0.5

            # plt.figure(figsize = (6,6))
            # plt.imshow(orig_img)
            # file_name = "input{:03}_noise.png".format(i)
            # plt.savefig(save_filepath+"/"+file_name)
            # plt.show()
            # plt.clf()
            # plt.close()

    return file_names


def main(load_filepath, sub_dir, save_filepath="afm_dataset", write_all_index=False, file_name="{:05}.png", 
    n_burst=10, sift_step_size = 1, kernel_size = 8,n_patch=2):
    files = glob.glob(load_filepath+"/" + sub_dir + "/*")
    n_set = int(len(files)/n_burst)

    sift_flow = SiftFlowTorch(
        cell_size=2,
        step_size=sift_step_size,
        is_boundary_included=False,
        num_bins=8,
        cuda=True,
        fp16=True,
        return_numpy=False)

    # リストをreset
    os.makedirs(save_filepath + "/" + sub_dir, exist_ok=True)
    with open(save_filepath + "/" + sub_dir + "/sep_trainlist.txt", mode='w') as f:
        pass

    for i in range(n_set):
        #index = list(range(i*n_burst, (i+1)*n_burst))
        index = files[i*n_burst:(i+1)*n_burst]
        imgs = read_imgs(load_filepath+"/"+sub_dir, file_name, index)
        patches = make_patch(imgs,n_patch=n_patch)
        for j, patch in enumerate(patches):
            flows = cal_flow(sift_flow, patch, kernel_size=kernel_size)
            noise_level_list = cal_noise_level(flows)
            file_names = save_img_and_flow(j, patch, noise_level_list, 
                        flows, save_filepath+"/"+sub_dir+"/set{:04}".format(i), visualize=True)

            # 21xxxxの中のindex
            with open(save_filepath + "/" + sub_dir + "/sep_trainlist.txt", mode='a') as f:
                for n  in range(len(file_names)):
                    if n or i or j:
                        f.write("\n")
                    f.write(file_names[n])
            
            # afm_datasetの中のindex
            if write_all_index:
                try:
                    with open(save_filepath + "/sep_trainlist.txt", mode="x") as f:
                        flag = 1
                except FileExistsError:
                    flag = 0
                
                with open(save_filepath+"/sep_trainlist.txt", mode='a') as f:
                    for n  in range(len(file_names)):
                        if not flag:
                            f.write("\n")
                        else:
                            flag = 0
                        f.write(sub_dir+"/"+file_names[n])

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
n_patch = 4
load_filename = "dirty"
save_filepath = f"dirty_dataset{n_patch}"

# main(load_filename, "20211109_2", save_filepath=save_filepath, kernel_size=7, n_patch=n_patch,write_all_index=False)

#main("orig_img", "210923", save_filepath=f"afm_dataset{n_patch}", kernel_size=7, n_patch=n_patch,write_all_index=True)
# main(load_filename, "20211022", save_filepath=save_filepath, kernel_size=7, n_patch=n_patch,write_all_index=True)
# main(load_filename, "20211023", save_filepath=save_filepath, kernel_size=7, n_patch=n_patch,write_all_index=True)
main(load_filename, "20211029", save_filepath=save_filepath, kernel_size=7, n_patch=n_patch,write_all_index=True)
main(load_filename, "20211030", save_filepath=save_filepath, kernel_size=7, n_patch=n_patch,write_all_index=True)
# main(load_filename, "20211106", save_filepath=save_filepath, kernel_size=7, n_patch=n_patch,write_all_index=True)
# main(load_filename, "20211108", save_filepath=save_filepath, kernel_size=7, n_patch=n_patch,write_all_index=True)
# main(load_filename, "20211109", save_filepath=save_filepath, kernel_size=7, n_patch=n_patch,write_all_index=True)
# main(load_filename, "20211112", save_filepath=save_filepath, kernel_size=7, n_patch=n_patch,write_all_index=True)
# main(load_filename, "20211115", save_filepath=save_filepath, kernel_size=7, n_patch=n_patch,write_all_index=True)
# main(load_filename, "20211116", save_filepath=save_filepath, kernel_size=7, n_patch=n_patch,write_all_index=True)
# main(load_filename, "20211122", save_filepath=save_filepath, kernel_size=7, n_patch=n_patch,write_all_index=True)
# main(load_filename, "20211126", save_filepath=save_filepath, kernel_size=7, n_patch=n_patch,write_all_index=True)

# main(load_filename, "20211126", save_filepath=save_filepaths, kernel_size=7, n_patch=n_patch,write_all_index=false)

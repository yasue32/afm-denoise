import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
import pyflow
from skimage import img_as_float
from random import randrange
import os.path
import cv2

import net_utils
import torchvision.transforms as transforms
import time

def get_flow(im1, im2):
    im1 = np.array(im1)
    im2 = np.array(im2)
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.
    
    # Flow Options:
    alpha = 0.007
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    
    u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,nSORIterations, colType)
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    #flow = rescale_flow(flow,0,1)
    return flow

# @brief AKAZEによる画像特徴量取得
# @param img 特徴量を取得したい画像（RGB順想定）
# @param pt1 特徴量を求める開始座標 tuple (default 原点)
# @param pt2 特徴量を求める終了座標 tuple (default None=画像の終わり位置)
# @return key points
def get_keypoints(img, pt1 = (0, 0), pt2 = None):
    if pt2 is None:
        pt2 = (img.shape[1], img.shape[0])
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = cv2.rectangle(np.zeros_like(gray), pt1, pt2, color=1, thickness=-1)
    #sift = cv2.AKAZE_create()
    sift = cv2.SIFT_create()
    # find the key points and descriptors with AKAZE
    return sift.detectAndCompute(gray, mask=mask)

# @brief imgと、特徴記述子kp2/des2にマッチするような pointを求める
# @param img 特徴量を取得したい画像（RGB順想定）
# @param kp2 ベースとなる画像のkeypoint
# @param des2 ベースとなる画像の特徴記述
# @return apt1 imgの座標 apt2 それに対応するkp2
def get_matcher(img, kp2, des2):
    kp1, des1 = get_keypoints(img)
    if len(kp1) == 0 or len(kp2) == 0:
        return None,None    

    # Brute-Force Matcher生成
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    if len(matches[0])==1:
        return None, None
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) == 0:
        return None,None
    target_position = []
    base_position = []
    # x,y座標の取得
    for g in good:
        target_position.append([kp1[g.queryIdx].pt[0], kp1[g.queryIdx].pt[1]])
        base_position.append([kp2[g.trainIdx].pt[0], kp2[g.trainIdx].pt[1]])

    apt1 = np.array(target_position)
    apt2 = np.array(base_position)
    return apt1, apt2

# @brief マッチング画像生成(kp2にマッチするような画像へimgを変換)
# @param img 変換させる画像（RGB順想定）
# @param kp2 ベースとなる画像のkeypoint
# @param des2 ベースとなる画像の特徴記述
# @return アフィン変換後の画像 (行列推定に失敗するとNoneが返る)
def get_alignment_img(img, kp2, des2):
 
    height, width = img.shape[:2]
    # 対応点を探索
    apt1, apt2 = get_matcher(img, kp2, des2)
    if type(apt1)==type(apt2):
        return None
 
    # アフィン行列の推定
    mtx = cv2.estimateAffinePartial2D(apt1, apt2)[0]
 
    # アフィン変換
    if mtx is not None:
        return cv2.warpAffine(img, mtx, (width, height))
    else:
        return None

# target: numpy.array
# input: [np1, np2, ...]  
# def flow_align(target, input, device):
#     loader = transforms.Compose([transforms.ToTensor()]) 
#     flow_torch = torch.empty(len(input), 2, *input[0].shape[:2]).to(device)
#     input_torch = torch.empty(len(input), *input[0].shape).to(device)
#     for i, img in enumerate(input):
#         tmp1 = torch.from_numpy(get_flow(target, img)).unsqueeze(dim=0).permute(0,3,1,2).float().to(device)
#         flow_torch[i] = tmp1
#         input_torch[i] = loader(img).unsqueeze(dim=0).float().to(device)
    
#     warped_input_torch = net_utils.backward_warp(input_torch, flow_torch)
#     warped_input_list = []
#     for i in range(len(flow_torch)):
#         warped_input_list.append(np.array(warped_input_torch[i].permute(1,2,0)))
    
#     return warped_input_list

def affine_align(target, input):
    aligned_list=[]
    kp, des = get_keypoints(np.array(target))
    if len(kp)==0:
        return input
    for img in input:
        img2 = np.array(img)    
        align = get_alignment_img(img2, kp, des)
        if type(align)==type(None):
            align = img2.copy()
        aligned_list.append(align)
    return aligned_list
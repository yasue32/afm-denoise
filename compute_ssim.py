from subprocess import PIPE
import subprocess
import os
from glob import glob
from copy import deepcopy
import pandas as pd 
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class NoiseScore():
    def __init__(self, ch=3):
        kernel = torch.Tensor([[-1., 0., 1.],
                               [-2., 0., 2.],
                               [-1., 0., 1.]])
        # [output_ch, input_ch, kernelH, kernelW]
        self.kernelx = kernel.reshape(1, 1, *kernel.shape).repeat(ch, ch, 1, 1)
        self.kernely = torch.transpose(self.kernelx, 2, 3)

    def __call__(self, imgs):
        scores = {}
        imgs = self.array2Tensor(imgs)
        scores = self.calc(imgs)
        scores_sorted = {key: val for key, val in list(enumerate(scores))}
        scores_sorted = np.array(
            sorted(scores_sorted.items(), key=lambda x: x[1], reverse=True))

        return scores_sorted[:, 0].astype(np.uint8), scores
        # return scores_sorted[:, 0], scores_sorted[:, 1] # idxs, scores

    def calc(self, imgs):
        print(imgs.shape)
        score_map = (F.conv2d(imgs, self.kernely).abs() -
                     F.conv2d(imgs, self.kernelx).abs())
        # [B, C, H, W] -> [B], * Divide 3 is an operation to match the score when applied to a single image read by L type.
        scores = score_map.mean(dim=(1, 2, 3)) / 3
        return scores.to('cpu').detach().numpy()

    def array2Tensor(self, imgs):
        # [B, H, W, C] -> [B, C, H, W]
        return torch.from_numpy(imgs).permute(0, 3, 1, 2)

class BlurScore():
    def __init__(self):
        pass

    def __call__(self, imgs):
        scores = {}
        # print(imgs.shape)
        for idx, img in enumerate(imgs):
            scores[idx] = self.calc(img)

        scores_sorted = np.array(
            sorted(scores.items(), key=lambda x: x[1], reverse=True))
        return scores_sorted[:, 0].astype(np.uint8), np.array(list(scores.values()))
        # return scores_sorted[:, 0], scores_sorted[:, 1] # idxs, scores

    def calc(self, img):
        # print(img.shape, type(img))
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        # print(type(img), img, img.shape)
        return cv2.Laplacian((img * 255).astype(np.uint8), cv2.CV_64F).var()

filenames = [
    "Results/pretrained2x_mse_Pflow_f1_256"
]

# n_frames = 4

sub = "test_dataset_per_sequence/sep_trainlist_1x"

target_name = "{}.png"
pred_name = "{}_RBPNF1.png"
inputs_name = "{}_inputs.png"

dataset = "test_dataset_per_sequence"


noise_filter = NoiseScore()
blur = BlurScore()
diff = 0
zouka = 0

for filename in filenames:
    path = filename + "/" + sub + "/[0-9]*.png"  
    files = glob(path)
    print(path, len(files)/3 )
    l = int(len(files)/3)
    columns = ["ssim_fake", "cw-ssim_fake", "ssim_baseframe", "cw-ssim_baseframe", "noise_target", "noise_fake", "noise_baseframe","blur_target", "blur_fake", "blur_baseframe"]
    index = np.arange(l)
    df = pd.DataFrame(columns=columns, index=index)

    for i in range(l):
        # target_file = "pretrained2x_mae_Pflow/test_dataset_per_sequence/sep_trainlist_2x/0_inputs.png"
        target_file = filename + "/" + sub + "/" + target_name.format(i)
        pred_file = filename + "/" + sub + "/" + pred_name.format(i)
        inputs_file = filename + "/" + sub + "/" + inputs_name.format(i)

        target_image = cv2.imread(target_file).astype(np.float32) 
        pred_image = cv2.imread(pred_file).astype(np.float32) 
        inputs_image = cv2.imread(inputs_file).astype(np.float32)
        command = "pyssim"
        option = "--cw"
        # print(command)
        
        # dataset_sub = str(i).zfill(4)
        # dataset_imgs = glob(dataset + "/" + dataset_sub + "/*")
        inputs_ssim = 0
        inputs_ssim_cw = 0
        h_inputs = len(inputs_image)
        w_inputs = len(inputs_image[0])
        # for k in range(int(w_inputs/h_inputs) - 1):
        input_image = inputs_image[:,int(h_inputs*0):int(h_inputs*1)]
        input_file = filename + "/" + sub + "/base_" + inputs_name.format(i)
        cv2.imwrite(input_file, input_image)
        base_ssim_cw = subprocess.run([command, option, target_file, input_file] ,stdout=PIPE, stderr=PIPE)
        base_ssim = subprocess.run([command, target_file, input_file] ,stdout=PIPE, stderr=PIPE)

        comp_process1 = subprocess.run([command, option, target_file, pred_file] ,stdout=PIPE, stderr=PIPE)
        comp_process2 = subprocess.run([command, target_file, pred_file] ,stdout=PIPE, stderr=PIPE)
        noise_inds, noise_scores = noise_filter(deepcopy(np.stack([target_image, pred_image, input_image])/255))
        blur_inds, blur_scores = blur(deepcopy(np.stack([target_image, pred_image, input_image])))
        # comp_process = subprocess.run("pyssim",stdout=PIPE, stderr=PIPE)
        print(i)
        print("cw-ssim", float(comp_process1.stdout))
        print("ssim",  float(comp_process2.stdout))
        # print(noise_scores)
        df["cw-ssim_fake"][i] = float(comp_process1.stdout)
        df["ssim_fake"][i] = float(comp_process2.stdout)
        df["ssim_baseframe"][i] = float(base_ssim.stdout)
        df["cw-ssim_baseframe"][i] = float(base_ssim_cw.stdout)
        df["noise_target"][i] = noise_scores[0]
        df["noise_fake"][i] = noise_scores[1]
        df["noise_baseframe"][i] = noise_scores[2]
        df["blur_target"][i] = blur_scores[0]
        df["blur_fake"][i] = blur_scores[1]
        df["blur_baseframe"][i] = blur_scores[2]

        diff += noise_scores[0] - noise_scores[1]
        if noise_scores[0] < noise_scores[1]:
            zouka += 1
        print("mean of diff:",diff/(i+1)) 
        print("ノイズ増加回数 {}/{},  {}%".format(zouka, i+1, zouka/(i+1)))       


        print(df.head)

        # print(comp_process.stderr)

    df.to_csv(filename + "/" + sub + "/eval.csv")
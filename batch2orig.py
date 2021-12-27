import argparse

import os
import cv2
import numpy as np
import time
import glob
from PIL import Image

n_batch = 4
n_burst = 10

load_filepath = "./Results"
sub = "/scratch1x_patch4_warping/afm_dataset4/211109_2/sep_trainlist_1x"

files = glob.glob(load_filepath + sub + "/*_RBPNF7.png")
save_filepath = "./concat"
os.makedirs(save_filepath + sub, exist_ok=True)

#print(files)
n = 0
batch = []
for i in range(0, len(files), n_burst-1):
    #print(n)
    n += 1
    batch.append(files[i])
    if (n % (n_batch**2)) == 0: 
        tmp = []
        for v in range(n_batch):
            file_list = batch[v*n_batch : (v+1)*n_batch]
            print(file_list)
            im_list = [cv2.imread(path) for path in file_list]
            tmp.append(np.concatenate(im_list, 1))
        im = np.concatenate(tmp, 0)
        batch = []
        pil_img = Image.fromarray(im)
        pil_img.save(save_filepath + sub + "/concat{:05}".format(n//(n_batch**2) - 1) + ".png")


            
            


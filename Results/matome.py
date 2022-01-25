import matplotlib.pyplot as plt 
from tqdm import tqdm 
from glob import glob 
import os 
import numpy as np
import cv2

filenames = [
            # "pretrained2x_mse_Pflow_test",
            # "pretrained2x_mse_Nflow_test",
            "pretrained2x_mae_Pflow", 
            # "pretrained2x_mae_Nflow",
            #"pretrained2x_mse_Pflow_f1",
            "pretrained2x_mae_Pflow_blur3",
            "pretrained2x_mae_Pflow_patch32",
            "pretrained2x_mae_Pflow_patch96",
            "pretrained2x_mae_Pflow_warping",
            #"pretrained2x_mse_Sflow_blur3_test",
            # "pretrained2x_mse_Pflow_blur3_Aloss015_3test",
            # "pretrained2x_mse_Pflow_blur3_Aloss025_3test",
            # "pretrained2x_mse_Pflow_blur3_Aloss035_3test",
            # "pretrained2x_mse_Pflow_f4",
            # "pretrained2x_mse_Pflow_res_blur3",
            # "pretrained2x_mse_Pflow_blur3_Aloss015",
            # "pretrained2x_mse_Pflow_blur3_Aloss025",
            # "pretrained2x_mse_Pflow_blur3_Aloss035",
            # "pretrained2x_mse_Pflow_blur3_Aloss045",
            ]

subs = ["test_dataset_per_sequence/sep_trainlist_2x",
        "test_dataset_per_sequence/sep_trainlist_2x",
        "test_dataset_per_sequence/sep_trainlist_2x",
        "test_dataset_per_sequence/sep_trainlist_2x",
        #"afm_dataset4/20211109_2/sep_trainlist_1x",
        # "test_dataset_per_sequence/sep_trainlist_1x",
        "test_dataset_per_sequence/sep_trainlist_2x",
        "test_dataset_per_sequence/sep_trainlist_2x",
        "test_dataset_per_sequence/sep_trainlist_2x",
        # "test_dataset_per_sequence/sep_trainlist_2x",
]
print(len(subs), len(filenames))

w = 4
h = int(len(filenames)/4) + 2


# sub = "dirty_dataset4/sep_trainlist_2x"

save_filepath = "matome/0122"

l = 2000
fake_path = []
for i, filename in enumerate(filenames):
    #sub = "afm_dataset4/20211109_2/sep_trainlist_2x"
    path = filename + "/" + subs[i] + "/[0-9]*.png"
    if "f4" in filename:
        fake_path.append("{}_RBPNF4.png")    
    elif "f1" in filename:
        fake_path.append("{}_RBPNF1.png")
        #sub = "afm_dataset4/20211109_2/sep_trainlist_1x"
        # path = filename + "/" + subs[i] + "/*"  
    else:
        fake_path.append("{}_RBPNF7.png")
    #subs.append(sub)
    #os.makedirs(save_filepath + "/" + subs[i], exist_ok=True)
    files = glob(path)
    print(filename, len(files))
    l = min(l, int(len(files)/3))


# print(fake_path)

files = ""

gt_path = "{}.png"
inputs_path = "{}_inputs.png"

fs = 6

for i in tqdm(range(l)):
    gt = cv2.imread(filenames[0] + "/" + subs[0] + "/" + gt_path.format(i))
    # print(filenames[0] + "/" + subs[0] + "/" + gt_path.format(i))
    assert gt is not None
    plt.figure(figsize=(10,10))
    plt.title(str(i))
    plt.subplot(h, w, 1+w)
    plt.title("GT",  fontsize=fs)
    plt.axis('off')
    plt.imshow(gt)
    inputs = cv2.imread(filenames[0] + "/" + subs[0] + "/" + inputs_path.format(i))
    h_inputs = len(inputs)
    w_inputs = len(inputs[0])
    # print(h_inputs, w_inputs)
    for k in range(int(w_inputs/h_inputs) - 1 ):
        plt.subplot(h, 7, k+1)
        plt.title("input "+str(k), fontsize=fs)
        plt.axis('off')
        plt.imshow(inputs[:,int(h_inputs*k):int(h_inputs*(k+1))])
    num_img = 1 + 1 + w
    for j in range(len(filenames)):
        img = cv2.imread(filenames[j] + "/" + subs[j] + "/" + fake_path[j].format(i))
        assert img is not None
        plt.subplot(h, w, num_img)
        plt.axis('off')
        plt.imshow(img)
        plt.title(filenames[j], fontsize=fs)
        num_img += 1
    os.makedirs(save_filepath + "/test/" + subs[0], exist_ok=True)
    plt.savefig(save_filepath + "/test/" + subs[0] + f"/{i}.png",bbox_inches='tight')
    plt.clf()
    plt.close()
import os
import glob
import shutil

import torch
import torch.nn.functional as F
from sift_flow_torch import SiftFlowTorch
from third_party.flowiz import flowiz
import cv2
from tqdm import tqdm

device = device=torch.device('cuda')
LOAD_DIR = "ext_clean_dataset_per_sequence"

seqs = glob.glob(os.path.join(LOAD_DIR, "*"))


for seq in seqs:
    files = glob.glob(os.path.join(seq, "*.pt"))
    for index, filepath in enumerate(files):
        flows = torch.load(filepath) 
        print(flows.shape)
        print(flows[0][0][5])            
import glob
import os
from copy import deepcopy
from random import Random
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

random.seed(346)

class RealDataset(Dataset):
    def __init__(self, opt):
        pass

    def __getitem__(self, ind):
        pass


    def __len__(self):
        return len(self.seq)



class SynthDataset(Dataset):
    def __init__(self, opt):
        pass

    def __getitem__(self, ind):
        pass


    def __len__(self):
        return len(self.seq)

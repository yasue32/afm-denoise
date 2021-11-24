from os.path import exists, join, basename
from os import makedirs, remove
import os
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, ToTensor

from dataset import DatasetFromFolderTest, DatasetFromFolder

def transform():
    return Compose([
        ToTensor(),
    ])

# modified by shinjo 1120
def get_training_set(data_dir, nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size, future_frame, shuffle, upscale_only):
    print("Training samples chosen:", file_list)
    return DatasetFromFolder(data_dir,nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size,future_frame, shuffle,
                             transform=transform(), upscale_only=upscale_only)


def get_eval_set(data_dir, nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size, future_frame, upscale_only):
    return DatasetFromFolder(data_dir,nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size,future_frame,
                             transform=transform(), upscale_only=upscale_only)

def get_test_set(data_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame, upscale_only):    
    return DatasetFromFolderTest(data_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame, transform=transform(), upscale_only=upscale_only)


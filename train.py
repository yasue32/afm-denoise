import argparse
import gc
import os
import random
import sys
from copy import deepcopy
from email import generator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim

import utils
# from dataset_real import RealDataset, SynthDataset
from burstsr_dataset import BurstSRDataset as RealDataset
from rrdbnet_arch import RRDBNet

parser = argparse.ArgumentParser(description='Train Real-burst-ESRGAN')
parser.add_argument('-o', '--options', type=str, default=None, help="import .yaml file")
args = parser.parse_args()

try:
    with open(args.options) as file:
        opt = yaml.safe_load(file)
except Exception as e:
    print('Exception occurred while loading YAML...', file=sys.stderr)
    print(e, file=sys.stderr)
    sys.exit(1)


def trainModel(
        epoch,
        training_data_loader,
        Generator,
        optimizerG,
        GeneratorCriterion,
        device,
        opt):
    # torch.autograd.set_detect_anomaly(True)
    trainBar = tqdm(training_data_loader)
    runningResults = {'batchSize': 0, 'GLoss': 0}
    Generator.train()

    # skip first-iter
    iterTrainBar = iter(trainBar)
    next(iterTrainBar)

    runningResults['batchSize'] += opt["train"]["batch_size"]

    if opt["train"]["parallel"]:
        G_parameters = list(Generator.module.parameters())
    else:   
        G_parameters = list(Generator.parameters())

    for data in iterTrainBar:
        burst, frame_gt, meta_info_burst, meta_info_gt = data[0], data[1], data[2], data[3]
        check = burst

        burst = Variable(burst).to(device)
        frame_gt = Variable(frame_gt).to(device)

        fakeHR = Generator(burst)
        # print(torch.sum(torch.abs(fakeHR-frame_gt)))

        Generator.zero_grad()
        # print(burst.shape, frame_gt.shape, fakeHR.shape)
        GLoss = GeneratorCriterion(frame_gt, fakeHR)
        GLoss.backward(inputs=G_parameters)
        optimizerG.step()

        runningResults['GLoss'] = GLoss.item() * runningResults["batchSize"]
        trainBar.set_description(
            desc='[Epoch: %d/%d] G Loss: %.4f' %
                (epoch, opt["train"]["end_epoch"],runningResults['GLoss'] / runningResults['batchSize']))

        # gc.collect()
    return runningResults


def main():
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and opt["gpu_mode"] else "cpu")

    if opt["input_type"]=="real":
        train_set = RealDataset(opt, burst_size=opt["dataset"]["burst_size"])
    elif opt["input_type"] == "synth":
        train_set = SynthDataset(opt)
    
    training_data_loader = DataLoader(
        dataset=train_set,
        num_workers=opt["train"]["threads"],
        batch_size=opt["train"]["batch_size"],
        shuffle=True)
    # print(len(train_set))

    Generator = RRDBNet(**opt["net"], scale=opt["upscale_factor"])
    
    if opt["train"]["parallel"]:
        gpus_list = list(range(torch.cuda.device_count()))
        Generator = torch.nn.DataParallel(Generator, device_ids=gpus_list)

    GeneratorCriterion = nn.L1Loss()
    if torch.cuda.is_available() and opt["gpu_mode"]:
        utils.printCUDAStats()
        Generator = Generator.to(device)
        GeneratorCriterion = GeneratorCriterion.to(device)

    optimizerG = optim.Adam(
        Generator.parameters(), lr=opt["train"]["lr"], betas=(
            0.9, 0.99), eps=1e-8)
    
    if opt["use_wandb"]:
        import wandb
        wandb.init(project="BurstSR", config=opt)
        wandb.watch(Generator)

    utils.printNetworkArch(Generator, None)

    if opt["train"]["pretrained"]:
        modelPath = os.path.join(opt["train"]["save_folder"] + opt["train"]["pretrained_g"])
        utils.loadPreTrainedModel(
            gpuMode=opt["gpu_mode"],
            model=Generator,
            modelPath=modelPath,
            device=device)

    os.makedirs(opt["train"]["save_folder"], exist_ok=True)

    # --- Running
    for epoch in range(opt["train"]["start_epoch"], opt["train"]["end_epoch"] + 1):
        runningResults = trainModel(
            epoch,
            training_data_loader,
            Generator,
            optimizerG,
            GeneratorCriterion,
            device,
            deepcopy(opt))
        # if args.use_wandb:
        #     wandb.log(runningResults)

        # if (epoch + 1) % (args.snapshots) == 0:
        if epoch % opt["train"]["save_model_per"] == 0:  # shinjo
            saveModelParams(
                epoch,
                runningResults,
                Generator,
                opt["train"]["save_folder"],
                opt["upscale_factor"])


def saveModelParams(
        epoch,
        runningResults,
        netG,
        save_folder,
        upscale_factor):
    pathG = os.path.join(
        save_folder, 'g_x%d_%d.pth' %(upscale_factor, epoch))
    pathD = os.path.join(
        save_folder, 'd_x%d_%d.pth' %(upscale_factor, epoch))
    
    results = {'GLoss': [], 'GScore': [], 'PSNR': [], 'SSIM': []}

    # Save model parameters
    torch.save(netG.state_dict(), pathG)


if __name__ == "__main__":
    main()

import argparse
import gc
import os

import pandas as pd
# import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
from copy import deepcopy

import logger
import utils
# from data import get_training_set
from dbpns import Net as DBPNS
from rbpn import GeneratorLoss
from rbpn import Net as RBPN
from rbpn import Net2 as RBPN2
from SRGAN.model import Discriminator
from UNet import UNet16

# ################################################# iSEEBETTER TRAINER KNOB
# upscale_factor = 4
# #########################################################################

# Handle command line arguments
parser = argparse.ArgumentParser(description='Train iSeeBetter: Super Resolution Models')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=5, help='testing batch size')
parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--gpu_id', default="", type=str, help='not using')
parser.add_argument('--data_dir', type=str, default='./vimeo_septuplet/sequences')
parser.add_argument('--file_list', type=str, default='sep_trainlist.txt')
parser.add_argument('--other_dataset', type=bool, default=False, help="use other dataset than vimeo-90k")
parser.add_argument('--future_frame', type=bool, default=True, help="use future frame")
parser.add_argument('--nFrames', type=int, default=7)
parser.add_argument('--patch_size', type=int, default=64, help='0 to use original frame size')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--model_type', type=str, default='RBPN', help="RBPN or U-Net")
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--pretrained_sr', default='RBPN_4x.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='F7', help='Location to save checkpoint models')
parser.add_argument('--APITLoss', action='store_true', help='Use APIT Loss')
parser.add_argument('--useDataParallel', action='store_true', help='Use DataParallel')
parser.add_argument('-v', '--debug', default=False, action='store_true', help='Print debug spew.')

parser.add_argument('--RBPN_only', action='store_true', required=False, help="use RBPN only")
parser.add_argument('--log_dir', type=str, default="./log", help="location to save log ")
parser.add_argument('--shuffle', action='store_true', required=False, help="Use shuffle dataset")
parser.add_argument('--denoise', action='store_true', required=False, help="set --upscalefactor 1 and --pretreined model")
parser.add_argument('--warping', action='store_true', required=False, help="warping input imgs to target")
parser.add_argument('--alignment', action='store_true', required=False, help="alignment input imgs to target")
parser.add_argument('--use_wandb', action='store_true', required=False, help="use wandb logger")
parser.add_argument('--use_tensorboard', action='store_true', required=False, help="use tensorboard logger")
parser.add_argument('--num_channels', type=int, default=3, help="channels of img")
parser.add_argument('--depth_img', action='store_true', required=False, help="when use depth(numpy.npy) img")
parser.add_argument('--optical_flow', type=str, default="p", help="p=pyflow, n=noting")
parser.add_argument('--pretrained_d', default="", help='pretrained Discriminator model')
parser.add_argument('--noise_flow_type', type=str, default="p", help="s=sift_flow, p=filter")
# parser.add_argument('--random_crop', type=int, default=0, help='0 to use original frame size (for AFM project)')
parser.add_argument("--aloss", default=0.005, help="weights of adversarial loss")
# parser.add_argument('--network_type', type=str, default="RBPN", help="type of generator network. RBPN or UNet")


def trainModel(
        epoch,
        training_data_loader,
        netG,
        netD,
        optimizerD,
        optimizerG,
        generatorCriterion,
        device,
        args):
    torch.autograd.set_detect_anomaly(True)

    trainBar = tqdm(training_data_loader)
    runningResults = {
        'batchSize': 0,
        'DLoss': 0,
        'GLoss': 0,
        'DScore': 0,
        'GScore': 0}

    netG.train()
    if netD:  # shinjo modified
        netD.train()

    # Skip first iteration
    iterTrainBar = iter(trainBar)
    next(iterTrainBar)

    if args.useDataParallel:
        G_parameters = list(netG.module.parameters())
        if netD:
            try:
                D_parameters = list(netD.module.parameters())
            except BaseException:
                D_parameters = list(netD.parameters())

    else:
        G_parameters = list(netG.parameters())
        if netD:
            try:
                D_parameters = list(netD.parameters())
            except BaseException:
                D_parameters = list(netD.module.parameters())

    if args.model_type == "UNet":
        unet_repeat = 9
    else:
        unet_repeat = 1

    for data in iterTrainBar:
        batchSize = len(data)
        runningResults['batchSize'] += batchSize

        #######################################################################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        #######################################################################

        # if args.APITLoss:
        #     fakeHRs = []
        #     targets = []
        fakeScrs = []
        realScrs = []

        DLoss = 0

        for repeater in range(unet_repeat):

            input, target, neigbor, flow, bicubic, good_img = data[0], data[1], data[2], data[3], data[4], data[5]
            if args.gpu_mode and torch.cuda.is_available():
                input = Variable(input).to(device)  # shinjo modified
                target = Variable(target).to(device)
                bicubic = Variable(bicubic).to(device)
                neigbor = [Variable(j).to(device) for j in neigbor]
                realHR = Variable(good_img).to(device)
                flow = [Variable(j).to(device).float() for j in flow]
            else:
                input = Variable(input).to(device=device, dtype=torch.float)
                target = Variable(target).to(device=device, dtype=torch.float)
                bicubic = Variable(bicubic).to(device=device, dtype=torch.float)
                neigbor = [
                    Variable(j).to(
                        device=device,
                        dtype=torch.float) for j in neigbor]
                realHR = Variable(realHR).to(device=device, dtype=torch.float)
                flow = [Variable(j).to(device=device, dtype=torch.float)
                        for j in flow]

            if args.model_type == "UNet":
                if repeater == (unet_repeat - 1):
                    fakeHR = netG(input)
                else:
                    # print("neigbor", neigbor[repeater].shape, input.shape)
                    fakeHR = netG(neigbor[repeater])
                # length = len(neigbor)
                # target = sum(neigbor)
                # if length > 0:
                #     target /= length
                # fakeHR = netG(input)
                # for j in range(len(neigbor[0])):
                #     tmp = torch.zeros((0, *target.shape[-3:])).to(target.device)
                #     for i in range(length):
                #         tmp = torch.cat((tmp, neigbor[i][j].unsqueeze(dim=0)))
                #     fakeHR = torch.cat((fakeHR, netG(tmp)))
                # target = torch.cat([deepcopy(target)] * (length+1))

            else:
                if args.nFrames == 1:
                    fakeHR = netG(input)
                else:
                    fakeHR = netG(input, neigbor, flow)

            if args.residual:
                fakeHR = fakeHR + bicubic

            if netD:
                netD.zero_grad()

                # realOut = netD(target).mean()
                # print(realHR.shape, fakeHR.shape, target.shape)
                realOut = netD(realHR).mean()
                fakeOut = netD(fakeHR).mean()
                DLoss = (fakeOut - realOut)

                DLoss.backward(retain_graph=True, inputs=D_parameters)

            # fakeHR [B, C, H, W] (0~1)
            # affine変換したときの補完領域でlossを計算しない
            # if args.alignment:
            #     mask = torch.ones(target.shape[0], 1, target.shape[2], target.shape[3]).to(device=device)
            #     mask_color = [0,1,0]
            #     for channel in range(target.shape[1]):
            #         mask = mask * (target[:,channel]==mask_color[channel])
                # print(mask)


            netG.zero_grad()
            if args.APITLoss:
                assert netD is not None
                fakeOut = netD(fakeHR).mean()
                GLoss = generatorCriterion(fakeOut, fakeHR, target, 0)
            else:
                assert netD is None

                GLoss = generatorCriterion(fakeHR, target)

            GLoss.backward(inputs=G_parameters)
            if netD is not None:
                optimizerD.step()
            optimizerG.step()

            if netD is not None:
                realScrs.append(realOut)
                fakeScrs.append(fakeOut)

            # fakeHR = netG(input, neigbor, flow)
            # if args.residual:
            #     fakeHR = fakeHR + bicubic

            # if netD: # shinjo modified
            #     realOut = netD(target).mean()
            #     DLoss_real = (0.5 - realOut) / len(data)
            #     # DLoss_real.backward(retain_graph = True)

            #     fakeOut = netD(fakeHR).mean()
            #     DLoss_fake = (0.5 + fakeOut) / len(data)
            #     # DLoss_fake.backward(retain_graph = False)

            #     DLoss = DLoss_fake + DLoss_real
            #     DLoss.backward()

            #     # realOut = netD(target).mean()
            #     # fakeOut = netD(fakeHR).mean()

            #     # DLoss_real = -realOut
            #     # DLoss_fake = fakeOut

            #     if args.APITLoss:
            #         fakeHRs.append(fakeHR)
            #         targets.append(target)
            #     fakeScrs.append(fakeOut)
            #     realScrs.append(realOut)

            #     # DLoss = DLoss + 1 - realOut + fakeOut
            #     # DLoss = DLoss / len(data)

            #     # Calculate gradients

            #     # DLoss = DLoss_fake + DLoss_real
            #     # DLoss.backward(retain_graph=True) # default: True

            #     # Update weights
            #     optimizerD.step()

            #     fakeSrcs = []
            #     fakeOut = netD(fakeHR).mean()
            #     fakeScrs.append(fakeOut)

            # ################################################################################################################
            # # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            # ################################################################################################################
            # GLoss = 0

            # # Zero-out gradients, i.e., start afresh
            # netG.zero_grad()

            # if args.APITLoss:
            #     if not netD: # shinjo modified
            #         raise Exception("If you use APITLoss, please run without --RBPN_only")
            #     idx = 0

            #     for fakeHR, fake_scr, HRImg in zip(fakeHRs, fakeScrs, targets):
            #         fakeHR = fakeHR.to(device)
            #         fake_scr = fake_scr.to(device)
            #         HRImg = HRImg.to(device)
            #         GLoss += generatorCriterion(fake_scr, fakeHR, HRImg, idx)
            #         idx += 1
            # else:
            #     GLoss = generatorCriterion(fakeHR, target)

            # GLoss = GLoss / len(data)

            # # Calculate gradients
            # GLoss.backward()

            # # Update weights
            # optimizerG.step()

            runningResults['GLoss'] += GLoss.item() * args.batchSize

            if netD:
                realOut = torch.Tensor(realScrs).mean()
                fakeOut = torch.Tensor(fakeScrs).mean()
                runningResults['DLoss'] += DLoss.item() * args.batchSize
                runningResults['DScore'] += realOut.item() * args.batchSize
                runningResults['GScore'] += fakeOut.item() * args.batchSize

                trainBar.set_description(
                    desc='[Epoch: %d/%d] G Loss: %.4f' %
                    (epoch, args.nEpochs, runningResults['GLoss'] / runningResults['batchSize']))
            else:
                trainBar.set_description(
                    desc='[Epoch: %d/%d] D Loss: %.4f G Loss: %.4f D(x): %.4f D(G(z)): %.4f' %
                    (epoch,
                    args.nEpochs,
                    runningResults['DLoss'] /
                    runningResults['batchSize'],
                    runningResults['GLoss'] /
                    runningResults['batchSize'],
                    runningResults['DScore'] /
                    runningResults['batchSize'],
                    runningResults['GScore'] /
                    runningResults['batchSize']))
            gc.collect()

    netG.eval()

    # learning rate is decayed by a factor of 10 every half of total epochs
    if (epoch + 1) % (args.nEpochs / 2) == 0:
        for param_group in optimizerG.param_groups:
            param_group['lr'] /= 10.0
        logger.info(
            'Learning rate decay: lr=%s',
            (optimizerG.param_groups[0]['lr']))

    return runningResults


def saveModelParams(
        epoch,
        runningResults,
        netG,
        netD,
        save_folder,
        upscale_factor,
        writer):
    pathG = os.path.join(
        save_folder, 'netG_epoch_%d_%d.pth' %
        (upscale_factor, epoch))  # shinjo
    pathD = os.path.join(
        save_folder, 'netD_epoch_%d_%d.pth' %
        (upscale_factor, epoch))
    if netD:  # shinjo modified
        results = {
            'DLoss': [],
            'GLoss': [],
            'DScore': [],
            'GScore': [],
            'PSNR': [],
            'SSIM': []}

        # Save model parameters
        torch.save(netG.state_dict(), pathG)
        torch.save(netD.state_dict(), pathD)

        logger.info("Checkpoint saved to {}".format(pathG))
        logger.info("Checkpoint saved to {}".format(pathD))

        # Save Loss\Scores\PSNR\SSIM
        results['DLoss'].append(
            runningResults['DLoss'] /
            runningResults['batchSize'])
        results['GLoss'].append(
            runningResults['GLoss'] /
            runningResults['batchSize'])
        results['DScore'].append(
            runningResults['DScore'] /
            runningResults['batchSize'])
        results['GScore'].append(
            runningResults['GScore'] /
            runningResults['batchSize'])
        # results['PSNR'].append(validationResults['PSNR'])
        # results['SSIM'].append(validationResults['SSIM'])

        # tensorboard に保存
        if writer:
            writer.add_scalar("DLoss", results["DLoss"][-1], epoch)
            writer.add_scalar("GLoss", results["GLoss"][-1], epoch)
            writer.add_scalar("DScore", results["DScore"][-1], epoch)
            writer.add_scalar("GScore", results["GScore"][-1], epoch)

        if epoch % 1 == 0 and epoch != 0:
            out_path = 'statistics/'
            data_frame = pd.DataFrame(data={'DLoss': results['DLoss'], 'GLoss': results['GLoss'], 'DScore': results['DScore'],
                                            'GScore': results['GScore']},  # , 'PSNR': results['PSNR'], 'SSIM': results['SSIM']},
                                      index=range(1, epoch + 1))
            data_frame.to_csv(
                out_path +
                'iSeeBetter_' +
                str(upscale_factor) +
                '_Train_Results.csv',
                index_label='Epoch')
    else:
        results = {'GLoss': [], 'GScore': [], 'PSNR': [], 'SSIM': []}

        # Save model parameters
        torch.save(netG.state_dict(), pathG)

        logger.info("Checkpoint saved to {}".format(pathG))

        # Save Loss\Scores\PSNR\SSIM
        results['GLoss'].append(
            runningResults['GLoss'] /
            runningResults['batchSize'])
        results['GScore'].append(
            runningResults['GScore'] /
            runningResults['batchSize'])

        if writer:
            writer.add_scalar("GLoss", results["GLoss"][-1], epoch)
            writer.add_scalar("GScore", results["GScore"][-1], epoch)

        if epoch % 1 == 0 and epoch != 0:
            out_path = 'statistics/'
            data_frame = pd.DataFrame(
                data={
                    'GLoss': results['GLoss'],
                    'GScore': results['GScore']},
                index=range(
                    1,
                    epoch + 1))
            data_frame.to_csv(
                out_path +
                'iSeeBetter_' +
                str(upscale_factor) +
                '_Train_Results.csv',
                index_label='Epoch')


def main():
    """ Lets begin the training process! """

    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    gpus_list = list(range(args.gpus))

    # device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available()
    # and args.gpu_mode else "cpu") # shinjo modified
    device = torch.device("cuda" if torch.cuda.is_available(
    ) and args.gpu_mode else "cpu")  # shinjo modified
    if args.gpu_id != "":
        print("*** use CUDA_VISIBLE_DEVICES={} *** ".format(args.gpu_id))

    # Initialize Logger
    logger.initLogger(args.debug)

    # Load dataset
    logger.info('==> Loading datasets')
    # train_set = get_training_set(args.data_dir, args.nFrames, args.upscale_factor, args.data_augmentation,
    #                              args.file_list, args.other_dataset, args.patch_size, args.future_frame,
    # args.shuffle, (args.denoise), args.warping, args.alignment,
    # args.depth_img, args.optical_flow, args.random_crop)

    from dataset_akita import TrainDataset
    train_set = TrainDataset(args.data_dir, args.nFrames, 
        train=True, noise_flow_type=args.noise_flow_type, optical_flow=args.optical_flow, patch_size=args.patch_size, warping=args.warping)

    training_data_loader = DataLoader(
        dataset=train_set,
        num_workers=args.threads,
        batch_size=args.batchSize,
        shuffle=True)

    # Use generator as RBPN
    if args.model_type == "UNet":
        netG = UNet16(num_classes=3, num_filters=32, 
            pretrained=args.pretrained, up_sampling_method="deconv")
    elif args.model_type == "RBPN":
        if args.nFrames == 1:
            netG = DBPNS(base_filter=3, feat=3, num_stages=3,
                        scale_factor=args.upscale_factor)  # 20211208
        else:
            if args.denoise:
                if not args.depth_img:
                    netG = RBPN2(
                        num_channels=args.num_channels,
                        base_filter=256,
                        feat=64,
                        num_stages=3,
                        n_resblock=5,
                        nFrames=args.nFrames,
                        scale_factor=args.upscale_factor)
                else:
                    netG = RBPN2(
                        num_channels=1,
                        base_filter=256,
                        feat=64,
                        num_stages=3,
                        n_resblock=5,
                        nFrames=args.nFrames,
                        scale_factor=args.upscale_factor)
            else:
                netG = RBPN(
                    num_channels=args.num_channels,
                    base_filter=256,
                    feat=64,
                    num_stages=3,
                    n_resblock=5,
                    nFrames=args.nFrames,
                    scale_factor=args.upscale_factor)
    logger.info('# of Generator parameters: %s', sum(param.numel()
                for param in netG.parameters()))

    # Use DataParallel?
    if args.useDataParallel:
        netG = torch.nn.DataParallel(netG, device_ids=gpus_list)

    # Use discriminator from SRGAN
    if not args.RBPN_only:  # shinjo modified
        netD = Discriminator()
        if args.pretrained_d != "":
            modelPath = os.path.join(args.save_folder + args.pretrained_d)
            utils.loadPreTrainedModel(
                gpuMode=args.gpu_mode,
                model=netD,
                modelPath=modelPath,
                device=device)
        logger.info('# of Discriminator parameters: %s',
                    sum(param.numel() for param in netD.parameters()))
    else:
        netD = None

    # Generator loss
    generatorCriterion = nn.L1Loss() if not args.APITLoss else GeneratorLoss(args.aloss)

    if args.gpu_mode and torch.cuda.is_available():
        utils.printCUDAStats()
        netG = netG.to(device)  # shinjo modified
        if not args.RBPN_only:  # shinjo modified
            netD = netD.to(device)

        generatorCriterion = generatorCriterion.to(device)

    # Use Adam optimizer
    optimizerG = optim.Adam(
        netG.parameters(), lr=args.lr, betas=(
            0.9, 0.999), eps=1e-8)
    if not args.RBPN_only:  # shinjo modified
        optimizerD = optim.Adam(
            netD.parameters(), lr=args.lr, betas=(
                0.9, 0.999), eps=1e-8)
    else:
        optimizerD = None

    if args.APITLoss:
        logger.info(
            "Generator Loss: Adversarial Loss + Perception Loss + Image Loss + TV Loss")
    else:
        logger.info("Generator Loss: L1 Loss")

    # print iSeeBetter architecture
    utils.printNetworkArch(netG, netD)

    # W&B
    if args.use_wandb:
        import wandb
        wandb.init(project="afm-image_denoising", config=args, name=args.save_folder.split("/")[-2])
        wandb.watch(netG)

    # tensorboard
    if args.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.log_dir)
    else:
        writer = None

    if args.pretrained:
        modelPath = os.path.join(args.save_folder + args.pretrained_sr)
        utils.loadPreTrainedModel(
            gpuMode=args.gpu_mode,
            model=netG,
            modelPath=modelPath,
            device=device)

    os.makedirs(args.save_folder, exist_ok=True)

    for epoch in range(args.start_epoch, args.nEpochs + 1):
        runningResults = trainModel(
            epoch,
            training_data_loader,
            netG,
            netD,
            optimizerD,
            optimizerG,
            generatorCriterion,
            device,
            args)
        if args.use_wandb:
            wandb.log(runningResults)

        # if (epoch + 1) % (args.snapshots) == 0:
        if epoch % args.snapshots == 0:  # shinjo
            # saveModelParams(epoch, runningResults, netG, netD, args.save_folder)
            saveModelParams(
                epoch,
                runningResults,
                netG,
                netD,
                args.save_folder,
                args.upscale_factor,
                writer)
    if args.use_tensorboard:
        writer.close()


if __name__ == "__main__":
    # mp.set_start_method('spawn')
    main()

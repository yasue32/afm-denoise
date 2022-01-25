import glob
import itertools
import os
from copy import deepcopy
from random import Random
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import net_utils
import torchvision.transforms as transforms

import pyflow
from sift_flow_torch import SiftFlowTorch
from third_party.flowiz import flowiz

random.seed(346)


class TrainDataset(Dataset):
    def __init__(self, data_dir, nFrames, train=True, noise_flow_type="p", optical_flow="p", patch_size=64, warping=False):

        if train:
            # self.sequence_dirs = [os.path.join(
            #     data_dir, x) for x in os.listdir(data_dir)]
            self.sequence_dirs = [
                [os.path.join(data_dir, x)] * 16 for x in os.listdir(data_dir)]
            self.sequence_dirs = list(
                itertools.chain.from_iterable(self.sequence_dirs))

            
            self.crop = RandomCrop(size=int(patch_size))
            self.noisy_inputs = False

            if noise_flow_type == "s":  # s as sift-flow
                self.noise_threshold = 0.01
                self.blur_threshold_lower = 250
                self.blur_threshold_upper = 1500
            elif noise_flow_type == "p":  # p as pyflow but means filter mode
                self.noise_threshold = 0.02
                self.blur_threshold_lower = 250
                self.blur_threshold_upper = 1500
            else:
                self.noise_threshold = 0.02
        else:
            print("TEST ....")
            bunkatsu = int(256//patch_size)
            self.sequence_dirs = [
                [os.path.join(data_dir, x)] *(bunkatsu**2) for x in os.listdir(data_dir)]
            self.sequence_dirs = list(
                itertools.chain.from_iterable(self.sequence_dirs))
            # print(self.sequence_dirs)
            self.crop = RasterCrop(size=int(patch_size))
            self.noise_threshold = 2
            self.blur_threshold_lower = 0
            self.blur_threshold_upper = 100000
        
            # testでもGTを選択する場合
            if noise_flow_type == "s":  # s as sift-flow
                self.noise_threshold = 9000
                self.blur_threshold_lower = 250
                self.blur_threshold_upper = 1500
            elif noise_flow_type == "p":  # p as pyflow but means filter mode
                self.noise_threshold = 0.05
                self.blur_threshold_lower = 250
                self.blur_threshold_upper = 1500
            
            # デモンストレーション用
            # self.noisy_inputs = True


        self.calc_scores = DegradationScore(noise_flow_type=noise_flow_type)

        self.noise_flow_type = noise_flow_type
        self.optical_flow = optical_flow
        # self.noise_threshold = 0.02
        # self.blur_threshold_lower = 100
        # self.blur_threshold_upper = 1000
        self.nFrames = nFrames
        self.train = train
        self.patch_size = int(patch_size)
        self.warping = warping

        self.seq_imgs = np.array([])
        self.seq_flows = np.array([])


    def __getitem__(self, index):
        # print(index)
        images, flows = self._load_images(index)

        images, target_ind, input_ind, neigbor_inds, good_img_bools = self._crop(
            images, flows, self.sequence_dirs[index])
        # if images is None:
        #     return None, None, None, None, None
        # images = images[inds][::-1]
        if self.train:
            images = self._data_augmentation(deepcopy(images))


        target = images[target_ind]
        input = images[input_ind]
        neigbors = images[neigbor_inds]
        good_img = random.choice(images[good_img_bools])

        if self.train:
            np.random.shuffle(neigbors)
        neigbors = neigbors[:self.nFrames-1]

        bicubic = input
        # print(target_ind, input_ind, neigbor_inds)
        # print(target.shape, input.shape, neigbors.shape)
        if self.optical_flow == "n":
            flows = [np.zeros((self.patch_size,self.patch_size,2)) for _ in range(self.nFrames-1)]
        else:
            flows = self._get_flow(input, neigbors)

        flow = self._get_flow(input, [target])[0]
        
        input = self._to_tensor(input)
        target = self._to_tensor(target)
        good_img = self._to_tensor(good_img)
        neigbors = [self._to_tensor(x) for x in neigbors]
        bicubic = self._to_tensor(bicubic)
        flows = [self._to_tensor(x) for x in flows]
        
        if self.warping:
            warped_target, warped_neigbor = self._warping_img(target, input, neigbors, flows, flow)
            # print(input.shape, warped_input.shape, neigbors[0].shape, warped_neigbor[0].shape)
            if self.train:
                target = deepcopy(warped_target)
            neigbors = [deepcopy(im) for im in warped_neigbor]  
        # self.iter += 1
        # if self.iter%10 == 0:
        # print(*self.times.items())
        # print(neigbor.shape, flows.shape)
        return input, target, neigbors, flows, bicubic, good_img

    def _crop(self, images, flows, filepath):
        for _ in range(self.train*99 + 1):
            # print("\n\n", len(self.seq_imgs), "\n\n")
            # if self.noise_flow_type == "s":
            if self.train == False:
                if len(self.seq_imgs) == 0:
                    # print("new data loading")
                    self.seq_imgs, self.seq_flows = self.crop(
                        deepcopy(images), flows.clone(), self.noise_flow_type)
                # print(self.seq_imgs, self.seq_flows)
                # print(self.seq_imgs.shape)
                cropped_img, cropped_flow = self.seq_imgs[0], self.seq_flows[0]
                self.seq_imgs, self.seq_flows = self.seq_imgs[1:], self.seq_flows[1:]
                # print("\n\n", len(self.seq_imgs), "\n\n")
                (blur_inds, blur_scores), (noise_inds, noise_scores) = self.calc_scores(
                    cropped_img, cropped_flow, filepath)
                if self.noisy_inputs:
                    cropped_img = deepcopy(cropped_img[noise_inds[:8]])
                    images = deepcopy(images[noise_inds[:8]])
                    print(images.shape)
                    (blur_inds, blur_scores), (noise_inds, noise_scores) = self.calc_scores(
                        cropped_img, cropped_flow, filepath)
            
            elif self.train == True:
                cropped_img, cropped_flow = self.crop(deepcopy(images),np.zeros((10, 10, images[0].shape[0], images[0].shape[1], 2)), self.noise_flow_type)
                # print(cropped_img)
                (blur_inds, blur_scores), (noise_inds, noise_scores) = self.calc_scores(cropped_img, cropped_flow, filepath)


            # print(noise_inds)
            # print(noise_scores)
            # print(noise_inds[noise_scores<self.noise_threshold])

            non_noise_bools = noise_scores < self.noise_threshold
            non_blur_bools = (self.blur_threshold_lower < blur_scores) * \
                (blur_scores < self.blur_threshold_upper).T

            good_img_bools = non_noise_bools * non_blur_bools.T
            if sum(good_img_bools) == 0:
                continue

            target_ind = None
            target_noise_score = 100000
            input_ind = None
            input_noise_score = 100000
            neigbor_inds = []

            if sum(good_img_bools) >= 2:
                for i, im in enumerate(images):
                    if good_img_bools[i] and (target_noise_score > noise_scores[i]):
                        if input_ind is not None:
                            neigbor_inds.append(input_ind)
                        input_ind = target_ind
                        input_noise_score = target_noise_score
                        target_noise_score = noise_scores[i]
                        target_ind = i
                        continue
                    elif good_img_bools[i] and (input_noise_score > noise_scores[i]):
                        if input_ind is not None:
                            neigbor_inds.append(input_ind)
                        input_ind = i
                        input_noise_score = noise_scores[i]
                        continue
                    neigbor_inds.append(i)
            else:
                for i, im in enumerate(images):
                    if good_img_bools[i]:
                        target_noise_score = noise_scores[i]
                        target_ind = i
                        continue

                    if input_noise_score > noise_scores[i]:
                        if input_ind is not None:
                            neigbor_inds.append(input_ind)
                        input_noise_score = noise_scores[i]
                        input_ind = i
                        continue

                    neigbor_inds.append(i)


            # for i, im in enumerate(images):
            #     if good_img_bools[i] and target_noise_score > noise_scores[i]:
            #         target_ind = i
            #         continue

            #     if sum(good_img_bools) >= 2:
            #         if good_img_bools[i] and (input_ind is None):
            #             input_ind = i
            #             continue
            #     else:
            #         if input_ind is None:
            #             input_ind = i
            #             input_noise_score = noise_scores[i]
            #             continue
            #         elif input_noise_score > noise_scores[i]:
            #             neigbor_inds.append(input_ind)
            #             input_ind = i
            #             input_noise_score = noise_scores[i]
            #             continue
            #     neigbor_inds.append(i)
            # print(target_ind, input_ind, neigbor_inds)
            return cropped_img, target_ind, input_ind, neigbor_inds, good_img_bools

            # return cropped
            # non_noise_inds = noise_inds[noise_scores<self.noise_threshold]
            # if len(non_noise_inds) == 0:
            #     continue

            # if self.blur_threshold_upper > sorted(blur_scores[non_noise_inds])[0] > self.blur_threshold_lower:
            #     # print("noise:", noise_scores)
            #     # print("blur: ", blur_scores)
            #     # print(_)
            #     best_ind = blur_inds[non_noise_inds]
            #     return cropped, noise_inds

        # print('illegal sample was detected')
        input_ind = noise_inds[-2]
        target_ind = noise_inds[-1]
        neigbor_inds = [i for i in range(
            len(images))if i != input_ind and i != target_ind]
        good_img_bools = [False] * int(len(images))
        good_img_bools[target_ind] = True
        return cropped_img, target_ind, input_ind, neigbor_inds, good_img_bools

    def _get_flow(self, input, neigbors):
        # Flow Options:
        alpha = 0.012
        ratio = 0.75
        minWidth = 20
        nOuterFPIterations = 7
        nInnerFPIterations = 1
        nSORIterations = 30
        # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
        colType = 0

        flows = []
        for neigbor in neigbors:
            # print(input.shape, neigbor.shape, neigbor.mean())
            u, v, im2W = pyflow.coarse2fine_flow(input.astype(np.double), neigbor.astype(
                np.double), alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations, colType)
            flow = np.concatenate((u[..., None], v[..., None]), axis=2)\

            flows.append(flow)

        return flows

    def _load_images(self, index):
        images = []
        flows = []
        
        for path in glob.glob(os.path.join(self.sequence_dirs[index], '*.png')):
            image = cv2.imread(path)
            if image is None:
                print(path)
            image = cv2.imread(path).astype(np.float32) / 255
            images.append(image)

            # if self.noise_flow_type == "s":
            flow_fname = os.path.basename(path).split('.', 1)[0]
            path2 = os.path.join(
                ".", self.sequence_dirs[index], f'{flow_fname}.pt')
            flow = torch.load(path2, map_location=torch.device('cpu'))
            # print(flow)
            flows.append(flow)
        # print(len(images))
        # if self.noise_flow_type == "s":
        return np.stack(images), torch.stack(flows)
        # else:
            # return np.stack(images) , None

    def _data_augmentation(self, images, flip_h=True, rot=True, gamma=True):
        length = len(images)
        if random.random() < 0.5 and flip_h:
            for i in range(length):
                images[i] = np.fliplr(images[i])  # flip h

        if rot:
            # if random.random() < 0.5:
            #     for i in range(length):
            #         images[i] = np.rot90(images[i], 2)  # mirror
            if random.random() < 0.5:
                for i in range(length):
                   images[i] = np.flipud(images[i])  # flip v

        if gamma:
            gamma_val = (random.random()/10*3+0.8)
            for i in range(length): 
                # images[i] = u_gamma(np.array(images[i]))
                images[i] = np.clip(1.0 * (images[i] / 1.0)**(gamma_val), 0, 1)
        return images

    def _warping_img(self, target, input, neigbor, flows, flow):
        # imgを [np1, np2, ...] -> torch に変換し、warping
        # torch -> [np1, ...] に戻す
        loader = transforms.Compose([transforms.ToTensor()])
        # flow = self._get_flow(target, input)[0]
        flow = torch.from_numpy(flow).unsqueeze(dim=0).permute(0, 3, 1, 2).float()
        target = target.unsqueeze(dim=0).float()
        warped_target = net_utils.backward_warp(target, flow)
        
        flows_torch = torch.empty(len(neigbor), *flow.shape[1:])
        neigbor_torch = torch.empty(len(neigbor), *target.shape[1:])
        for i, img in enumerate(neigbor):
            flows_torch[i] = flows[i]
            neigbor_torch[i] = neigbor[i]
        # tmp = net_utils.backward_warp(neigbor_torch, flow_torch)
        warped_neigbor = net_utils.backward_warp(neigbor_torch, flows_torch)
        # warped_neigbor = []
        # for i in range(len(flow_torch)):
        #     warped_neigbor.append(np.array(tmp[i].permute(1, 2, 0)))
        return warped_target[0], warped_neigbor

    def _to_tensor(self, images):
        return torch.from_numpy(images.astype(np.float32)).permute(2, 0, 1)

    def __len__(self):
        return len(self.sequence_dirs)


class RandomCrop:
    def __init__(self, size=(64, 64)):
        # print(size, type(size))
        if type(size) == int:
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, images, flows, noise_flow_type):
        _, height, width, _ = images.shape
        left = np.random.randint(width - self.size[0] + 1)
        top = np.random.randint(height - self.size[1] + 1)
        right = left + self.size[0]
        bottom = top + self.size[1]

        if noise_flow_type == "p":
            return images[:, top:bottom, left:right, :], flows[:, :, :, top:bottom, left:right]
        elif noise_flow_type == "s":
            flow_patches = []
            flow_patch = [0]*len(images)
            # print("random crop", flows.shape)
            # for n3 in range(len(images)):  # n_burst枚数ずつimgがある
                # flow_patch[n3] = flows[n3][:, top:bottom, left:right, :]
            flow_patches = flows[:, :, :, top:bottom, left:right]
            # print("random crop", flows.shape, "->", flow_patches.shape)
            # flow_patches.append(flow_patch)
            return images[:, top:bottom, left:right, :], torch.Tensor(flow_patches)


class RasterCrop:
    def __init__(self, size=(64, 64)):
        if type(size) == int:
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, imgs, flows, noise_flow_type):
        n_patch = int(imgs[0].shape[0]//self.size[0])
        n_burst = len(imgs)
        # # もしn_patchで割り切れなかったらエラーにする
        # if (len(imgs[0]) % n_patch):
        #     print('Error: patch division faild', file=sys.stderr)
        #     sys.exit(1)
        # img_patch_size = int(len(imgs[0][0]) / n_patch)

        img_patches = []
        flow_patches = []

        for n1 in range(n_patch):  # y方向のpatch分割
            isy, igy = n1 * self.size[0], (n1+1) * self.size[0]
            for n2 in range(n_patch):  # x方向のpatch
                isx, igx = n2 * self.size[1], (n2+1) * self.size[1]
                img_patch = [0]*n_burst
                flow_patch = [0]*n_burst
                for n3 in range(n_burst):  # n_burst枚数ずつimgがある
                    img_patch[n3] = deepcopy(imgs[n3][isy:igy, isx:igx, :])
                    flow_patch[n3] = deepcopy(flows[n3][:, :, isy:igy, isx:igx])
                img_patches.append(img_patch)
                # print(flow_patch[0].shape)
                flow_patches.append(flow_patch)
        # print(flow_patches[0].shape)
        # return np.array(img_patches), torch.Tensor(flow_patches)
        return np.array(img_patches), flow_patches


class DegradationScore:
    def __init__(self, ch=3, noise_flow_type="p"):
        self.noise_flow_type = noise_flow_type
        self.blurScore = BlurScore()
        self.noiseScore = NoiseScore(ch=ch)
        self.noiseScore2 = NoiseScore_SIFT_Loaded()

    def __call__(self, imgs, flows, filepath):
        # print(imgs.shape, type(imgs))
        # inds, scores = self.noiseScore2(imgs, flows)
        if self.noise_flow_type == "p":
            return self.blurScore(imgs), self.noiseScore(imgs)
        elif self.noise_flow_type == "s":
            return self.blurScore(imgs), self.noiseScore2(imgs, flows, filepath)


class NoiseScore():
    def __init__(self, ch):
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
        score_map = (F.conv2d(imgs, self.kernely).abs() -
                     F.conv2d(imgs, self.kernelx).abs())
        # [B, C, H, W] -> [B], * Divide 3 is an operation to match the score when applied to a single image read by L type.
        scores = score_map.mean(dim=(1, 2, 3)) / 3
        return scores.to('cpu').detach().numpy()

    def array2Tensor(self, imgs):
        # [B, H, W, C] -> [B, C, H, W]
        return torch.from_numpy(imgs).permute(0, 3, 1, 2)


class NoiseScore_SIFT_Loaded():
    def __init__(self):
        pass
        # self.model = SiftFlowTorch(
        #     cell_size=1,
        #     step_size=2,
        #     is_boundary_included=True,
        #     num_bins=8,
        #     cuda=False,
        #     fp16=True,
        #     return_numpy=False)

        # # loggerオブジェクトの宣言
        # self.logger = getLogger("LogTest")
        # # loggerのログレベル設定(ハンドラに渡すエラーメッセージのレベル)
        # self.logger.setLevel(logging.DEBUG)
        # # loggerのログレベル設定(ハンドラに渡すエラーメッセージのレベル)
        # self.logger.setLevel(logging.DEBUG)
        # self.file_handler = FileHandler('sample03.log', 'a')
        # self.file_handler.setLevel(logging.DEBUG)
        # self.file_handler.setFormatter(handler_format)
        # # テキスト出力のhandlerをセット
        # self.logger.addHandler(file_handler)

    def __call__(self, imgs, flows, filepath):
        # flows = self.cal_flow(self.model, imgs, kernel_size=7)
        sorted_noise_level_list, noise_level_list = self.cal_noise_level(flows)
        scores = []
        sorted_inds = []
        # self.flow_img_save(noise_level_list, imgs, filepath)
        for i in range(len(noise_level_list)):
            scores.append(noise_level_list[i][0])
            sorted_inds.append(sorted_noise_level_list[i][1])
        return np.array(sorted_inds), np.array(scores)

    def cal_flow(self, model, imgs, kernel_size=8):
        l = len(imgs)

        il1, il2, il3 = imgs[0].shape
        #print(imgs, l, imgs[0].shape)
        descs = model.extract_descriptor(imgs)
        # print('Descriptor shape:', descs.shape)
        flow_test = self.find_local_matches(descs[0:1], descs[1:2])
        # print(flow_test.shape)
        l0, l1, l2 = flow_test.shape
        diff1 = il1 - l1
        diff2 = il2 - l2
        flows = torch.zeros(l, l, l0, l1 + diff1, l2 + diff2)  # .to("cuda")
        for img1 in range(l-1):
            for img2 in range(img1+1, l):
                matches = self.find_local_matches(
                    descs[img1:img1+1], descs[img2:img2+1], kernel_size=kernel_size)
                flows[img1, img2, :, int(
                    diff1/2):int(l1+diff1/2), int(diff2/2):int(l2+diff2/2)] = matches
                flows[img2, img1, :, int(
                    diff1/2):int(l1+diff1/2), int(diff2/2):int(l2+diff2/2)] = matches
        return flows

    def find_local_matches(self, desc1, desc2, kernel_size=7):
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

    def cal_noise_level(self, flows):
        # flow[10,10,2,128,128]を受け取って、ノイズレベルを返す
        # [[noise0, index0],[noise1, index1],...]
        n_burst = len(flows)
        # l1 = len(flows[0,0])
        # l2 = len(flows[0,0,0])
        # print(flows.shape)
        # flows = flows.permute(0, 1, 3, 4, 2)
        noise_level_list = []

        for j in range(n_burst):
            noise_level = 0
            for k in range(n_burst):
                if j == k:
                    continue
                vector_map = flows[j][k]  # flow img-j to img-k
                vector_map = vector_map  #.permute(2, 0, 1)
                # print(vector_map.shape)
                tmp = torch.var(vector_map[0], dim=1) + \
                    torch.var(vector_map[1], dim=1)

                noise_level += torch.sum(tmp).item()
            # logger.debug(f"{}_{j}: {noise_level}")
            noise_level_list.append([noise_level, j])
        sorted_noise_level_list = sorted(noise_level_list)
        return sorted_noise_level_list, noise_level_list

    def flow_img_save(self, noise_level_list, imgs, filepath):
        files = glob.glob(filepath+'/*')
        path = filepath + '/noise{:04}'.format(len(files)-20) + '.jpg'
        # flow_img = flowiz.convert_from_flow(flow)

        h = 3
        w = 4
        fig = plt.figure()
        for i, img in enumerate(imgs):
            fig.add_subplot(h, w, i+1)
            plt.title(str(int(noise_level_list[i][0])), fontsize=8)
            plt.imshow(img)
        plt.savefig(path)
        plt.clf()
        plt.close()


class NoiseScore_SIFT():
    def __init__(self):
        self.model = SiftFlowTorch(
            cell_size=1,
            step_size=2,
            is_boundary_included=True,
            num_bins=8,
            cuda=False,
            fp16=True,
            return_numpy=False)

    def __call__(self, imgs):
        flows = self.cal_flow(self.model, imgs, kernel_size=7)
        sorted_noise_level_list, noise_level_list = self.cal_noise_level(flows)
        scores = []
        sorted_inds = []
        for i in range(len(noise_level_list)):
            scores.append(noise_level_list[i][0])
            sorted_inds.append(sorted_noise_level_list[i][1])
        return sorted_inds, scores

    def cal_flow(self, model, imgs, kernel_size=8):
        l = len(imgs)

        il1, il2, il3 = imgs[0].shape
        #print(imgs, l, imgs[0].shape)
        descs = model.extract_descriptor(imgs)
        # print('Descriptor shape:', descs.shape)
        flow_test = self.find_local_matches(descs[0:1], descs[1:2])
        # print(flow_test.shape)
        l0, l1, l2 = flow_test.shape
        diff1 = il1 - l1
        diff2 = il2 - l2
        flows = torch.zeros(l, l, l0, l1 + diff1, l2 + diff2)  # .to("cuda")
        for img1 in range(l-1):
            for img2 in range(img1+1, l):
                matches = self.find_local_matches(
                    descs[img1:img1+1], descs[img2:img2+1], kernel_size=kernel_size)
                flows[img1, img2, :, int(
                    diff1/2):int(l1+diff1/2), int(diff2/2):int(l2+diff2/2)] = matches
                flows[img2, img1, :, int(
                    diff1/2):int(l1+diff1/2), int(diff2/2):int(l2+diff2/2)] = matches
        return flows

    def find_local_matches(self, desc1, desc2, kernel_size=7):
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

    def cal_noise_level(self, flows):
        # flow[10,10,2,128,128]を受け取って、ノイズレベルを返す
        # [[noise0, index0],[noise1, index1],...]
        n_burst = len(flows)
        l1 = len(flows[0, 0])
        l2 = len(flows[0, 0, 0])
        flows = flows.permute(0, 1, 3, 4, 2)
        noise_level_list = []

        for j in range(n_burst):
            noise_level = 0
            for k in range(n_burst):
                if j == k:
                    continue
                vector_map = flows[j, k]  # flow img-j to img-k
                vector_map = vector_map.permute(2, 0, 1)
                # print(vector_map.shape)
                tmp = torch.var(vector_map[0], dim=1) + \
                    torch.var(vector_map[1], dim=1)
                noise_level += torch.sum(tmp).item()
            noise_level_list.append([noise_level, j])
        sorted_noise_level_list = sorted(noise_level_list)

        return sorted_noise_level_list, noise_level_list


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


if __name__ == '__main__':
    from torch.autograd import Variable
    from torch.utils.data import DataLoader

    from rbpn import Net2 as RBPN2
    device = 'cuda'

    # dataset = TrainDataset('afm_dataset_per_sequence', 3)
    dataset = TrainDataset('ext_clean_dataset_per_sequence', 7, train=True, noise_flow_type="p", optical_flow="p", warping=True)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1)

    save_dir = 'threshold_dicision1'

    def save_torch_img(img, path):
        img = img.numpy().transpose(1, 2, 0)*255
        img = img.astype(np.uint8)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, img)

    i = 0
    for data in dataloader:
        input, target, neigbor, flow, bicubic = data[0], data[1], data[2], data[3], data[4]
        #print(i, input.shape[0])

        for b in range(input.shape[0]):
            neigbor_cp = [n[b] for n in neigbor]
            # print(neigbor_cp[0].shape, input[b].shape, target[b].shape)
            neigbor_cat = torch.cat([target[b]] + [input[b]] + neigbor_cp , 2)
            # all_cat = torch.cat([input[b]] + [target[b]] + neigbor[b])
            # save_torch_img(input[b], os.path.join(
            #     save_dir, str(i).zfill(4) + '.png'))
            save_torch_img(neigbor_cat, os.path.join(
                save_dir, str(i).zfill(4) + '_clean_dirty.png'))
            i += 1

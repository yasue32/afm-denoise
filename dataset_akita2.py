from random import Random
import numpy as np
import os
import glob
import cv2
from copy import deepcopy

import pyflow

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from sift_flow_torch import SiftFlowTorch
from third_party.flowiz import flowiz

file_name = None

class TrainDataset(Dataset):
    def __init__(self, data_dir, nFrames):
        self.sequence_dirs = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]

        self.random_crop = RandomCrop()
        self.calc_scores = DegradationScore()

        self.noise_threshold = 0.02
        self.blur_threshold_lower = 250
        self.blur_threshold_upper = 1500
        self.nFrames = nFrames

    def __getitem__(self, index):
        # print(index)
        images, flows = self._load_images(index)
        images, target_ind, input_ind, neigbor_inds = self._crop(images, flows, self._load_images(index))
        
        # if images is None:
        #     return None, None, None, None, None
    
        # images = images[inds][::-1]

        target = images[target_ind]
        input = images[input_ind]
        neigbors = images[neigbor_inds]
        np.random.shuffle(neigbors)
        neigbors = neigbors[:self.nFrames-1]
        bicubic = input
        # print(target_ind, input_ind, neigbor_inds)
        # print(target.shape, input.shape)
        flows = self._get_flow(input, neigbors)

        input = self._to_tensor(input)
        target = self._to_tensor(target)
        neigbors = [self._to_tensor(x) for x in neigbors]
        bicubic = self._to_tensor(bicubic)
        flows = [self._to_tensor(x) for x in flows]

        return input, target, neigbors, flows, bicubic

    def _crop(self, images, flows):
        for _ in range(100):
            cropped_img, cropped_flow = self.random_crop(deepcopy(images), flows.clone())
            (blur_inds, blur_scores), (noise_inds, noise_scores) = self.calc_scores(cropped_img, cropped_flow)

            # print(noise_inds)
            # print(noise_scores)
            # print(noise_inds[noise_scores<self.noise_threshold])

            non_noise_bools = noise_scores<self.noise_threshold
            non_blur_bools = (self.blur_threshold_lower < blur_scores) *  (blur_scores< self.blur_threshold_upper).T

            good_img_bools = non_blur_bools * non_blur_bools.T
            if sum(good_img_bools)==0:
                continue

            target_ind = None
            input_ind = None
            neigbor_inds = []

            for i,im in enumerate(images):
                if good_img_bools[i] and target_ind is None:
                    target_ind = i
                    continue
                if sum(good_img_bools)>=2:
                    if good_img_bools[i] and input_ind is None:
                        input_ind = i
                        continue
                else:
                    if input_ind is None:
                        input_ind = i
                        continue                
                neigbor_inds.append(i)
            
            return cropped, target_ind, input_ind, neigbor_inds

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
        input_ind = 0
        target_ind = 1
        neigbor_inds = [i for i in range(len(images))if i!=input_ind and i!=target_ind]
        return cropped, target_ind, input_ind, neigbor_inds


    def _get_flow(self, input, neigbors):
        # Flow Options:
        alpha = 0.012
        ratio = 0.75
        minWidth = 20
        nOuterFPIterations = 7
        nInnerFPIterations = 1
        nSORIterations = 30
        colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

        flows = []
        for neigbor in neigbors:
            # print(input.shape, neigbor.shape, neigbor.mean())
            u, v, im2W = pyflow.coarse2fine_flow(input.astype(np.double), neigbor.astype(np.double), alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,nSORIterations, colType)
            flow = np.concatenate((u[..., None], v[..., None]), axis=2)\

            flows.append(flow)

        return flows


    def _load_images(self, index):
        images = []
        flows = []
        for path in glob.glob(os.path.join(self.sequence_dirs[index], '*.png')):
            image = cv2.imread(path).astype(np.float32) / 255
            images.append(image)

            fname = os.path.basename(path).split('.', 1)[0]
            path2 = glob.glob(os.path.join(self.sequence_dirs[index], f'{flow_fname}.pt'))
            flow = torch.load(path2)
            flows.append(flow)

        return np.stack(images) # , torch.stack(flows)

    def _to_tensor(self, images):
        return torch.from_numpy(images.astype(np.float32)).permute(2, 0, 1)

    def __len__(self):
        return len(self.sequence_dirs)


class RandomCrop:
    def __init__(self, size=(64, 64)):
        if size == int:
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, images, flows):
        _, height, width, _ = images.shape
        left = np.random.randint(width - self.size[0] + 1)
        top = np.random.randint(height - self.size[1] + 1)
        right = left + self.size[0]
        bottom = top + self.size[1]

        return images[:, top:bottom, left:right, :], flows[:, top:bottom, left:right, :]

class DegradationScore:
    def __init__(self, ch=3):
        self.blurScore = BlurScore()
        self.noiseScore = NoiseScore(ch=ch)
        self.noiseScore2 = NoiseScore_SIFT_Loaded()

    def __call__(self, imgs,  flows, filepath):
        # inds, scores = self.noiseScore2(imgs, flows)
        return self.blurScore(imgs), self.noiseScore2(imgs,flows filepath)


class NoiseScore():
    def __init__(self, ch):
        kernel = torch.Tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]])
        self.kernelx = kernel.reshape(1, 1, *kernel.shape).repeat(ch, ch, 1, 1) # [output_ch, input_ch, kernelH, kernelW]
        self.kernely = torch.transpose(self.kernelx, 2, 3)

    def __call__(self, imgs):
        scores = {}
        imgs = self.array2Tensor(imgs)
        scores = self.calc(imgs)
        scores_sorted = {key: val for key, val in list(enumerate(scores))}
        scores_sorted = np.array(sorted(scores_sorted.items(), key=lambda x:x[1], reverse=True))

        return scores_sorted[:, 0].astype(np.uint8), scores
        # return scores_sorted[:, 0], scores_sorted[:, 1] # idxs, scores

    def calc(self, imgs):
        score_map = (F.conv2d(imgs, self.kernely).abs() - F.conv2d(imgs, self.kernelx).abs())
        scores = score_map.mean(dim=(1, 2, 3)) / 3 # [B, C, H, W] -> [B], * Divide 3 is an operation to match the score when applied to a single image read by L type.
        return scores.to('cpu').detach().numpy()

    def array2Tensor(self, imgs):
        return torch.from_numpy(imgs).permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]

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
        filepath = '/'.join(filepath.split('/')[:-1])
        sorted_noise_level_list, noise_level_list = self.cal_noise_level(imgs, flows)
        self.flow_img_save(noise_level_list, imgs, filepath)
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
        #print(flow_test.shape)
        l0, l1, l2 = flow_test.shape
        diff1 = il1 - l1
        diff2 = il2 - l2
        flows = torch.zeros(l, l, l0, l1 + diff1, l2 + diff2)#.to("cuda")
        for img1 in range(l-1):
            for img2 in range(img1+1, l):
                matches = self.find_local_matches(descs[img1:img1+1], descs[img2:img2+1], kernel_size=kernel_size)
                flows[img1, img2, :, int(diff1/2):int(l1+diff1/2), int(diff2/2):int(l2+diff2/2)] = matches
                flows[img2, img1, :, int(diff1/2):int(l1+diff1/2), int(diff2/2):int(l2+diff2/2)] = matches  
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

    def cal_noise_level(self, imgs, flows):
        #flow[10,10,2,128,128]を受け取って、ノイズレベルを返す
        # [[noise0, index0],[noise1, index1],...] 
        n_burst = len(flows)
        l1 = len(flows[0,0])
        l2 = len(flows[0,0,0])
        flows = flows.permute(0,1,3,4,2)
        noise_level_list = []

        for j in range(n_burst):
            noise_level = 0
            for k in range(n_burst):
                if j == k:
                    continue
                vector_map = flows[j,k]  # flow img-j to img-k
                vector_map = vector_map.permute(2,0,1)
                #print(vector_map.shape)
                tmp = torch.var(vector_map[0], dim=1) + torch.var(vector_map[1], dim=1)

                noise_level += torch.sum(tmp).item()
            logger.debug(f"{}_{j}: {noise_level}")
            noise_level_list.append([noise_level, j])
        sorted_noise_level_list = sorted(noise_level_list)
        return sorted_noise_level_list, noise_level_list

    def flow_img_save(self, noise_level_list, imgs, filepath):
        files = glob.glob(filepath+'/*')
        path = filepath + '/noise{:04}'.format( len(files)-20) + '.png'
        # flow_img = flowiz.convert_from_flow(flow)

        h = 3
        w = 4
        plt.figure()
        for i, img in enumerate(imgs);
            fig.add_subplot(h,w,i)
            plt.title(str(noise_level[i]))
            plt.imshow(img)
        plt.savefig(path)

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
        #print(flow_test.shape)
        l0, l1, l2 = flow_test.shape
        diff1 = il1 - l1
        diff2 = il2 - l2
        flows = torch.zeros(l, l, l0, l1 + diff1, l2 + diff2)#.to("cuda")
        for img1 in range(l-1):
            for img2 in range(img1+1, l):
                matches = self.find_local_matches(descs[img1:img1+1], descs[img2:img2+1], kernel_size=kernel_size)
                flows[img1, img2, :, int(diff1/2):int(l1+diff1/2), int(diff2/2):int(l2+diff2/2)] = matches
                flows[img2, img1, :, int(diff1/2):int(l1+diff1/2), int(diff2/2):int(l2+diff2/2)] = matches  
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
        #flow[10,10,2,128,128]を受け取って、ノイズレベルを返す
        # [[noise0, index0],[noise1, index1],...] 
        n_burst = len(flows)
        l1 = len(flows[0,0])
        l2 = len(flows[0,0,0])
        flows = flows.permute(0,1,3,4,2)
        noise_level_list = []

        for j in range(n_burst):
            noise_level = 0
            for k in range(n_burst):
                if j == k:
                    continue
                vector_map = flows[j,k]  # flow img-j to img-k
                vector_map = vector_map.permute(2,0,1)
                #print(vector_map.shape)
                tmp = torch.var(vector_map[0], dim=1) + torch.var(vector_map[1], dim=1)
                noise_level += torch.sum(tmp).item()
            noise_level_list.append([noise_level, j])
        sorted_noise_level_list = sorted(noise_level_list)

        return sorted_noise_level_list, noise_level_list


class BlurScore():
    def __init__(self):
        pass

    def __call__(self, imgs):
        scores = {}
        for idx, img in enumerate(imgs):
            scores[idx] = self.calc(img)
    
        scores_sorted = np.array(sorted(scores.items(), key=lambda x:x[1], reverse=True))
        return scores_sorted[:, 0].astype(np.uint8), np.array(list(scores.values()))
        # return scores_sorted[:, 0], scores_sorted[:, 1] # idxs, scores

    def calc(self, img):
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        # print(type(img), img, img.shape)
        return cv2.Laplacian((img * 255).astype(np.uint8), cv2.CV_64F).var()


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torch.autograd import Variable
    from rbpn import Net2 as RBPN2
    device = 'cuda'

    # dataset = TrainDataset('afm_dataset_per_sequence', 3)
    dataset = TrainDataset('afm_dataset_per_sequence', 3)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1)

    save_dir = 'threshold_dicision3'

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
            save_torch_img(input[b], os.path.join(save_dir, str(i).zfill(4) + '.png'))
            i += 1
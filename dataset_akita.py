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

class TrainDataset(Dataset):
    def __init__(self, data_dir, nFrames):
        self.sequence_dirs = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]

        self.random_crop = RandomCrop()
        self.calc_scores = DegradationScore()

        self.noise_threshold = 0.015
        self.blur_threshold_lower = 250
        self.blur_threshold_upper = 1500
        self.nFrames = nFrames

    def __getitem__(self, index):
        images = self._load_images(index)
        images, inds = self._crop(images)

        # if images is None:
        #     return None, None, None, None, None
    
        images = images[inds]

        target = images[0]
        input = images[1]
        neigbors = images[2:]
        np.random.shuffle(neigbors)
        neigbors = neigbors[:self.nFrames-1]
        bicubic = input
        flows = self._get_flow(input, neigbors)

        input = self._to_tensor(input)
        target = self._to_tensor(target)
        neigbors = [self._to_tensor(x) for x in neigbors]
        bicubic = self._to_tensor(bicubic)
        flows = [self._to_tensor(x) for x in flows]

        return input, target, neigbors, flows, bicubic

    def _crop(self, images):
        for _ in range(100):
            cropped = self.random_crop(deepcopy(images))
            (blur_inds, blur_scores), (noise_inds, noise_scores) = self.calc_scores(cropped)


            non_noise_inds = noise_inds[noise_scores<self.noise_threshold]
            if len(non_noise_inds) == 0:
                continue

            if self.blur_threshold_upper > sorted(blur_scores[non_noise_inds])[0] > self.blur_threshold_lower:
                # print("noise:", noise_scores)
                # print("blur: ", blur_scores)
                # print(_)
                return cropped, blur_inds

        # print('illegal sample was detected')
        return cropped, blur_inds

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
            u, v, im2W = pyflow.coarse2fine_flow(input.astype(np.double), neigbor.astype(np.double), alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,nSORIterations, colType)
            flow = np.concatenate((u[..., None], v[..., None]), axis=2)\

            flows.append(flow)

        return flows


    def _load_images(self, index):
        images = []
        for path in glob.glob(os.path.join(self.sequence_dirs[index], '*.png')):
            image = cv2.imread(path).astype(np.float32) / 255
            images.append(image)

        return np.stack(images)

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

    def __call__(self, images):
        _, height, width, _ = images.shape
        left = np.random.randint(width - self.size[0] + 1)
        top = np.random.randint(height - self.size[1] + 1)
        right = left + self.size[0]
        bottom = top + self.size[1]

        return images[:, top:bottom, left:right, :]

class DegradationScore:
    def __init__(self, ch=3):
        self.blurScore = BlurScore()
        self.noiseScore = NoiseScore(ch=ch)

    def __call__(self, imgs):
        return self.blurScore(imgs), self.noiseScore(imgs)


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

    save_dir = 'threshold_dicision2'

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
import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join, dirname
from PIL import Image, ImageOps
import random
import pyflow
from skimage import img_as_float
from random import randrange, sample
import os.path
from glob import glob

import net_utils
import torchvision.transforms as transforms
import time
import alignment

# added by shinjo 1120
GAMMA = 2.0
GAMMA_TBL = [int((x / 255.) ** (1. / GAMMA) * 255.) for x in range(256)]
u_gamma = np.frompyfunc(lambda x: GAMMA_TBL[x], 1, 1)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

# shinjo modified
def load_img(filepath, nFrames, scale, other_dataset, upscale_only=False):
    seq = [i for i in range(1, nFrames)]
    #random.shuffle(seq) #if random sequence
    if other_dataset:
        if upscale_only:
            target = Image.open(filepath).convert('RGB')
            input=target
        else:
            target = modcrop(Image.open(filepath).convert('RGB'),scale)
            input=target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        
        char_len = len(filepath)
        neigbor=[]

        for i in seq:
            index = int(filepath[char_len-7:char_len-4])-i
            file_name=filepath[0:char_len-7]+'{0:03d}'.format(index)+'.png'
            
            if os.path.exists(file_name):
                if upscale_only:
                    temp = Image.open(filepath[0:char_len-7]+'{0:03d}'.format(index)+'.png').convert('RGB')
                else:
                    temp = modcrop(Image.open(filepath[0:char_len-7]+'{0:03d}'.format(index)+'.png').convert('RGB'),scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
                neigbor.append(temp)
            else:
                # print('neigbor frame is not exist')
                temp = input
                neigbor.append(temp)
    else:
        if upscale_only:
            target = Image.open(join(filepath,'im'+str(nFrames)+'.png')).convert('RGB')
            input  = target
            neigbor = [Image.open(filepath+'/im'+str(j)+'.png').convert('RGB') for j in reversed(seq)]
        else:
            target = modcrop(Image.open(join(filepath,'im'+str(nFrames)+'.png')).convert('RGB'), scale)
            input = target.resize((int(target.size[0]/scale), int(target.size[1]/scale)), Image.BICUBIC)
            neigbor = [modcrop(Image.open(filepath+'/im'+str(j)+'.png').convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC) for j in reversed(seq)]
    
    return target, input, neigbor

# shinjo modified 1120
def load_img_future(filepath, nFrames, scale, other_dataset, shuffle, upscale_only):
    tt = int(nFrames/2)
    if other_dataset:
        if upscale_only:
            #target = Image.open(filepath).convert('RGB')
            #input = target
            gt_path = "/".join(filepath.split("/")[:-1]) + "/gt.png"
            target = Image.open(gt_path).convert('RGB')
            input = Image.open(filepath).convert('RGB')
        else:
            #target = modcrop(Image.open(filepath).convert('RGB'),scale)
            #input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
            gt_path = "/".join(filepath.split("/")[:-1]) + "/gt.png"
            target = Image.open(gt_path).convert('RGB')
            input = modcrop(Image.open(filepath).convert('RGB'),scale)
            input = input.resize((int(input.size[0]/scale),int(input.size[1]/scale)), Image.BICUBIC)
        
        char_len = len(filepath)
        neigbor=[]
        neigbor_index=[]
        if shuffle: # add by shinjo 1120
            split_path = filepath.split("/")
            split_path[-1] = split_path[-1][:-7] + "[0-9][0-9][0-9].png"
            neigbor_index = [nfilepath for nfilepath in glob("/".join(split_path)) if nfilepath != filepath]
            neigbor_index = sample(neigbor_index, min(nFrames-1, len(neigbor_index)))
            if upscale_only:
                neigbor = [Image.open(nfilepath).convert('RGB') for nfilepath in neigbor_index]
            else:
                neigbor = [modcrop(Image.open(nfilepath).convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC) for nfilepath in neigbor_index]
            neigbor.extend([input for i in range(nFrames - 1 - len(neigbor))])
            neigbor_index.extend([filepath for i in range(nFrames - 1 - len(neigbor))])
        else:
            if nFrames%2 == 0:
                seq = [x for x in range(-tt,tt) if x!=0] # or seq = [x for x in range(-tt+1,tt+1) if x!=0]
            else:
                seq = [x for x in range(-tt,tt+1) if x!=0]
            #random.shuffle(seq) #if random sequence
            for i in seq:
                index1 = int(filepath[char_len-7:char_len-4])+i
                file_name1=filepath[0:char_len-7]+'{0:03d}'.format(index1)+'.png'

                if os.path.exists(file_name1):
                    if upscale_only:
                        temp = Image.open(file_name1).convert('RGB')
                    else:
                        temp = modcrop(Image.open(file_name1).convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
                    neigbor.append(temp)
                    neigbor_index.append(file_name1)
                else:
                    # print('neigbor frame- is not exist')
                    temp=input
                    neigbor.append(temp)
                    neigbor_index.append(filepath)
            
    else:
        if upscale_only:
            target = Image.open(join(filepath,'im4.png')).convert('RGB')
            input = target
        else:
            target = modcrop(Image.open(join(filepath,'im4.png')).convert('RGB'),scale)
            input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        neigbor = []
        seq = [x for x in range(4-tt,5+tt) if x!=4]
        #random.shuffle(seq) #if random sequence
        for j in seq:
            neigbor.append(modcrop(Image.open(filepath+'/im'+str(j)+'.png').convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC))
    return target, input, neigbor, filepath, neigbor_index

def load_img_future_depth(filepath, nFrames, scale, other_dataset, shuffle, upscale_only):
    tt = int(nFrames/2)
    if other_dataset:
        if upscale_only:
            #target = Image.open(filepath).convert('RGB')
            #input = target
            gt_path = "/".join(filepath.split("/")[:-1]) + "/gt.npy"
            # target = Image.open(gt_path).convert('RGB')
            target = Image.fromarray(np.load(gt_path))
            input = Image.fromarray(np.load(filepath))
            # input = Image.open(filepath).convert('RGB')
        else:
            #target = modcrop(Image.open(filepath).convert('RGB'),scale)
            #input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
            gt_path = "/".join(filepath.split("/")[:-1]) + "/gt.npy"
            # target = Image.open(gt_path).convert('RGB')
            # input = modcrop(Image.open(filepath).convert('RGB'),scale)
            target = Image.fromarray(np.load(gt_path))
            input = Image.fromarray(np.load(filepath))
            input = Image.fromarray(input).resize((int(input.size[0]/scale),int(input.size[1]/scale)), Image.BICUBIC)
            # input = input.resize((int(input.size[0]/scale),int(input.size[1]/scale)), Image.BICUBIC)
        
        char_len = len(filepath)
        neigbor=[]
        neigbor_index=[]
        if shuffle: # add by shinjo 1120
            split_path = filepath.split("/")
            split_path[-1] = split_path[-1][:-7] + "[0-9][0-9][0-9].npy"
            neigbor_index = [nfilepath for nfilepath in glob("/".join(split_path)) if nfilepath != filepath]
            neigbor_index = sample(neigbor_index, min(nFrames-1, len(neigbor_index)))
            if upscale_only:
                neigbor = [Image.fromarray(np.load(nfilepath)) for nfilepath in neigbor_index]
            else:
                neigbor = [modcrop(Image.fromarray(np.load(nfilepath)), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC) for nfilepath in neigbor_index]
            neigbor.extend([input for i in range(nFrames - 1 - len(neigbor))])
            neigbor_index.extend([filepath for i in range(nFrames - 1 - len(neigbor))])
        else:
            if nFrames%2 == 0:
                seq = [x for x in range(-tt,tt) if x!=0] # or seq = [x for x in range(-tt+1,tt+1) if x!=0]
            else:
                seq = [x for x in range(-tt,tt+1) if x!=0]
            #random.shuffle(seq) #if random sequence
            for i in seq:
                index1 = int(filepath[char_len-7:char_len-4])+i
                file_name1=filepath[0:char_len-7]+'{0:03d}'.format(index1)+'.npy'

                if os.path.exists(file_name1):
                    if upscale_only:
                        #temp = Image.open(file_name1).convert('RGB')
                        temp = Image.fromarray(np.load(file_name1))
                    else:
                        # temp = modcrop(Image.open(file_name1).convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
                        temp = modcrop(Image.fromarray(np.load(file_name1))).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
                    neigbor.append(temp)
                    neigbor_index.append(file_name1)
                else:
                    # print('neigbor frame- is not exist')
                    temp=input
                    neigbor.append(temp)
                    neigbor_index.append(filepath)
            
    else:
        if upscale_only:
            target = Image.open(join(filepath,'im4.png')).convert('RGB')
            input = target
        else:
            target = modcrop(Image.open(join(filepath,'im4.png')).convert('RGB'),scale)
            input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        neigbor = []
        seq = [x for x in range(4-tt,5+tt) if x!=4]
        #random.shuffle(seq) #if random sequence
        for j in seq:
            neigbor.append(modcrop(Image.open(filepath+'/im'+str(j)+'.png').convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC))
    return target, input, neigbor, filepath, neigbor_index

def get_flow(im1, im2):
    im1 = np.array(im1)
    im2 = np.array(im2)
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.
    
    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    
    u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,nSORIterations, colType)
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    #flow = rescale_flow(flow,0,1)
    return flow

def get_sift_flow(input_filepath, neigbor_filepath, input, neigbor):
    flow = []
    char_len = len(input_filepath)
    # input_index = int(input_filepath[char_len-7:char_len-4])
    input_flows = torch.load(input_filepath[:-4]+".pt")
    for path in neigbor_filepath:
        char_len = len(path)
        index1 = int(path[char_len-7:char_len-4])
        tmp = input_flows[index1+1].numpy()
        flow.append(tmp)
    return flow

def warping_img(target, input, neigbor):
    # imgを [np1, np2, ...] -> torch に変換し、warping
    # torch -> [np1, ...] に戻す
    loader = transforms.Compose([transforms.ToTensor()]) 
    flow = get_flow(target, input)
    flow = torch.from_numpy(flow).unsqueeze(dim=0).permute(0,3,1,2).float()
    input = loader(input).unsqueeze(dim=0).float()
    warped_input = net_utils.backward_warp(input, flow)
    
    flow_torch = torch.empty(len(neigbor), *flow.shape[1:])
    neigbor_torch = torch.empty(len(neigbor), *input.shape[1:])
    for i, img in enumerate(neigbor):
        tmp = torch.from_numpy(get_flow(target, img)).unsqueeze(dim=0).permute(0,3,1,2).float()
        flow_torch[i] = tmp
        neigbor_torch[i] = loader(img).unsqueeze(dim=0).float()
    tmp = net_utils.backward_warp(neigbor_torch, flow_torch)
    warped_neigbor = []
    for i in range(len(flow_torch)):
        warped_neigbor.append(np.array(tmp[i].permute(1,2,0)))
    return np.array(warped_input[0].permute(1,2,0)), warped_neigbor

def rescale_flow(x,max_range,min_range):
    max_val = np.max(x)
    min_val = np.min(x)
    return (max_range-min_range)/(max_val-min_val)*(x-max_val)+max_range

def modcrop(img, modulo):
    (ih, iw) = img.size
    ih = ih - (ih%modulo)
    iw = iw - (iw%modulo)
    img = img.crop((0, 0, ih, iw))
    return img

def get_patch(img_in, img_tar, img_nn, patch_size, scale, nFrames, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale #if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy,ix,iy + ip, ix + ip))#[:, iy:iy + ip, ix:ix + ip]
    img_tar = img_tar.crop((ty,tx,ty + tp, tx + tp))#[:, ty:ty + tp, tx:tx + tp]
    img_nn = [j.crop((iy,ix,iy + ip, ix + ip)) for j in img_nn] #[:, iy:iy + ip, ix:ix + ip]
                
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, img_nn, info_patch

def augment(img_in, img_tar, img_nn, flip_h=True, rot=True, gamma=True):# modified by shinjo 1120
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False, 'gamma': False}
    
    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        img_nn = [ImageOps.flip(j) for j in img_nn]
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            img_nn = [ImageOps.mirror(j) for j in img_nn]
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            img_nn = [j.rotate(180) for j in img_nn]
            info_aug['trans'] = True

    if gamma: # modified by shinjo 1120
        img_in = Image.fromarray(u_gamma(np.array(img_in)).astype(np.uint8))
        img_tar = Image.fromarray(u_gamma(np.array(img_tar)).astype(np.uint8))
        img_nn = [Image.fromarray(u_gamma(np.array(j)).astype(np.uint8)) for j in img_nn]
        info_aug['gamma'] = True

    return img_in, img_tar, img_nn, info_aug
    
def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir,nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size, future_frame, shuffle, transform=None, upscale_only=True, warping=False, alignment=False, depth_img=False):
        super(DatasetFromFolder, self).__init__()
        alist = [line.rstrip() for line in open(join(image_dir,file_list))]
        #print(alist)
        self.image_filenames = [join(image_dir,x) for x in alist]
        #print(self.image_filenames)
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.other_dataset = other_dataset
        self.patch_size = patch_size
        self.shuffle = shuffle # added by shinjo 1120
        self.future_frame = future_frame
        self.upscale_only = upscale_only
        self.warping = warping
        self.alignment = alignment
        self.depth_img = depth_img

    def __getitem__(self, index):
        if self.future_frame:
            if not self.depth_img:
                target, input, neigbor, input_filepath, neigbor_filepath = load_img_future(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset, self.shuffle, self.upscale_only) # modified by shinjo 1120
            else:
                target, input, neigbor, input_filepath, neigbor_filepath = load_img_future_depth(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset, self.shuffle, self.upscale_only)
        else:
            target, input, neigbor = load_img(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset)

        if self.patch_size != 0:
            input, target, neigbor, _ = get_patch(input,target,neigbor,self.patch_size, self.upscale_factor, self.nFrames)
        
        if self.data_augmentation:
            input, target, neigbor, _ = augment(input, target, neigbor)

        bicubic = rescale_img(input, self.upscale_factor)

        #flow = [get_flow(input,j) for j in neigbor]
        flow = get_sift_flow(input_filepath, neigbor_filepath, input, neigbor)

        if self.alignment:
            # print(input.size, neigbor[0].size)
            input = alignment.affine_align(target, [input])[0]
            neigbor = alignment.affine_align(target, neigbor)
            # print(input.shape, neigbor[0].shape)

        if self.warping:
            warped_input, warped_neigbor = warping_img(target, input, neigbor)
            input = warped_input
            neigbor = warped_neigbor            

        if self.transform:
            target = self.transform(target)
            input = self.transform(input)
            bicubic = self.transform(bicubic)
            neigbor = [self.transform(j) for j in neigbor]
            flow = [torch.from_numpy(j.transpose(2,0,1)) for j in flow]

        return input, target, neigbor, flow, bicubic

    def __len__(self):
        return len(self.image_filenames)

class DatasetFromFolderTest(data.Dataset):
    def __init__(self, image_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame, transform=None, upscale_only=True, warping=False, alignment=False):
        super(DatasetFromFolderTest, self).__init__()
        alist = [line.rstrip() for line in open(join(image_dir,file_list))]
        self.image_filenames = [join(image_dir,x) for x in alist]
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.other_dataset = other_dataset
        self.future_frame = future_frame
        self.upscale_only = upscale_only
        self.warping = warping
        self.alignment = alignment

    def __getitem__(self, index):
        if self.future_frame:
            target, input, neigbor, input_filepath, neigbor_filepath = load_img_future(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset, False, self.upscale_only) # modified by shinjo 1120
        else:
            target, input, neigbor = load_img(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset, self.upscale_only)

        bicubic = rescale_img(input, self.upscale_factor)
            
        #flow = [get_flow(input,j) for j in neigbor]
        flow = get_sift_flow(input_filepath, neigbor_filepath, input, neigbor)

        if self.alignment:
            neigbor = alignment.affine_align(input, neigbor)

        if self.warping:
            _, warped_neigbor = warping_img(input, input, neigbor)
            # input = warped_input
            neigbor = warped_neigbor
        
        if self.transform:
            target = self.transform(target)
            input = self.transform(input)
            bicubic = self.transform(bicubic)
            neigbor = [self.transform(j) for j in neigbor]
            flow = [torch.from_numpy(j.transpose(2,0,1)) for j in flow]
        
        return input, target, neigbor, flow, bicubic
      
    def __len__(self):
        return len(self.image_filenames)

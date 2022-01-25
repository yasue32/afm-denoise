import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import random


from dataset_akita3 import TrainDataset
test_set = TrainDataset("ext_clean_dataset_per_sequence", 7,
                        train=False, noise_flow_type="s", optical_flow="p", patch_size=256, warping=False)

test_data_loader = DataLoader(
    dataset=test_set,
    num_workers=1,
    batch_size=1,
    shuffle=False)
# next(iterTrainBar)

human_best = [
    5, 7, 6, 8, 8, 9, 2, 5, 8, 4,
    8, 3, 9, 9, 9, 9, 6, 8, 9, 8,
    3, 8, 8, 9, 4, 4, 7, 7, 3, 8, 
    9, 8, 7, 5, 8, 1, 2, 9, 9, 7, 
    7, 1, 9, 3, 9, 9, 6, 9, 7, 8
]
tp = 0
fp = 0
with tqdm(test_data_loader) as trainBar:
    iterTrainBar = iter(trainBar)
    for i, data in enumerate(iterTrainBar):
        if len(human_best) < (i+1):
            break
        input, target, neigbors, flows, bicubic, good_img_bools, images = data[
            0], data[1], data[2], data[3], data[4], data[5], data[6]
        # print(good_img_bools)
        good_index = [j for j, b in enumerate(good_img_bools[0]) if b != False]
        print("set:{}, human_best:{}".format(i, human_best[i]), good_index)
        trainBar.postfix = ("set:{}, human_best:{}, machine:".format(i, human_best[i]) + ",".join([str(j) for j in good_index]))
        if human_best[i] in good_index:
            tp += 1
            fp -= 1
        fp += len(good_index)

print("Length:{} , Score:{}".format(len(human_best), tp))
print("Acc:{}%".format(tp/len(human_best)*100))
print("Precision:{}%".format(tp/(tp + fp)*100))

# logを間引く
import os
from glob import glob
from tqdm import tqdm

path = "scratch1x_patch4_warping/"
name_G = "netG_epoch_1_{}.pth"
name_D = "netD_epoch_1_{}.pth"
files_G = glob(path + name_G.format("*"))
files_D = glob(path + name_D.format("*"))
start = 1
end = 1000

print(start,end)
not_delete_per = 10


for i in tqdm(range(start, end)):
    # print(path + name_D.format(i))
    if i % not_delete_per == 0:
        continue
    if os.path.isfile(path + name_D.format(i)):
        os.remove(path + name_D.format(i))
    if os.path.isfile(path + name_G.format(i)):
        os.remove(path + name_G.format(i))

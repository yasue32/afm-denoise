import os
import glob
import shutil

DATA_DIR = 'orig_img'
SAVE_DIR = 'afm_dataset_per_sequence'

i = 0

for pathes in zip(*[iter(sorted(glob.glob(os.path.join(DATA_DIR, '**/*.png'))))]*10):
    assert pathes[0][-5] == '0'
    save_dir = os.path.join(SAVE_DIR, str(i).zfill(4))
    os.makedirs(save_dir, exist_ok=True)
    for index, path in enumerate(pathes):
        shutil.copy(path, os.path.join(save_dir, str(index).zfill(4) + '.png'))

    i += 1

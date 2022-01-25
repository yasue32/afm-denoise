import os
import argparse


import sift_flow_correlation
import calc_noise
import make_dataset

#sub = "210923"

parser = argparse.ArgumentParser()
parser.add_argument("--sub_dir", type=str, default="210923")
parser.add_argument("--sift_flow", action='store_true', required=False)
parser.add_argument("--calc_noise", action='store_true', required=False)
parser.add_argument("--make_dataset", action='store_true', required=False)
parser.add_argument("--make_index", action='store_true', required=False)
parser.add_argument("--gpu_id", type=int, default=99)
parser.add_argument("--n_batch", type=int, default=2)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


#filename = "AutoScan2.0_{:05}_1.spm.tif"
filename = "{:05}.png"
#filename = "5um CNT.0_{:05}_1.spm.tif"
sub_dirs = args.sub_dir.split(",")

for i, sub in enumerate(sub_dirs):
    if args.sift_flow:
        sift_flow_correlation.flow_main(
            "orig_img/"+sub, filename,"result_flow/"+sub, kernel_size=5)

    if args.calc_noise:
        calc_noise.cal_noise_level("result_flow/"+sub, "result_noise/"+sub, "orig_img/"+sub, v=True)

    if args.make_dataset:
        make_dataset.make_dataset(load_noise_filepath="result_noise/"+sub,load_img_filepath="orig_img/"+sub, 
                    load_img_filename=filename, save_filepath="dataset/"+sub,
                    n_batch=args.n_batch, n_burst=10, n_set=9, gamma_corr=True)

    # 全体を含むインデックスを作成
    if args.make_index:
        with open("dataset/sep_trainlist.txt", mode="a") as a:
            if i:
                a.writelines("\n")
            with open("dataset/"+sub+"/sep_trainlist.txt",mode="r") as p:
                for s_line in p:
                    a.writelines(sub+"/"+s_line)
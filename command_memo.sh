
# GPUの指定
CUDA_VISIBLE_DEVICES=x python3 iSeeBetter...


## train
python3 iSeeBetterTrain.py --upscale_factor 2 --RBPN_only --other_dataset True --data_dir afm_dataset_per_sequence --file_list sep_trainlist.txt --patch_size 64 --batchSize 12 --denoise --save_folder weights/pretrained2x_mse/ --gpus 2 --useDataParallel --pretrained --pretrained_sr RBPN_2x.pth --use_wandb --optical_flow p
python3 iSeeBetterTrain.py --upscale_factor 2 --APITLoss --other_dataset True --data_dir ext_clean_dataset_per_sequence --file_list sep_trainlist.txt --patch_size 64 --batchSize 12 --denoise --save_folder weights/pretrained2x_mse_Pflow_blur3_Aloss015_3/ --gpus 1 --pretrained --pretrained_sr netG_epoch_2_2400.pth --nFrames 7 --nEpochs 10000 --snapshots 150 --pretrained_d netD_epoch_2_2400.pth --start_epoch 2401 --use_wandb


## test
python3 iSeeBetterTest.py --upscale_factor 2 --other_dataset True --data_dir test_dataset_per_sequence --file_list sep_trainlist.txt --gpu_mode --gpu_id 0 --gpus 1 --model weights/pretrained2x_mae_Pflow_blur3/netG_epoch_2_1500.pth --upscale_only --denoise --output Results/pretrained2x_mae_Pflow_blur3/ --optical_flow p


## 再現実験
# ベースとなる実験
python3 iSeeBetterTrain.py --upscale_factor 2 --RBPN_only --other_dataset True --data_dir afm_dataset_per_sequence --file_list sep_trainlist.txt --patch_size 64 --batchSize 12 --denoise --save_folder weights/pretrained2x_mse_Pflow/ --gpus 2 --useDataParallel --pretrained --pretrained_sr RBPN_2x.pth --use_wandb --optical_flow p

# 位置合わせ（ワーピング）--warping

# 位置合わせ（アフィン変換) --data_dir xxx_dataset_per_sequence_aligned
python3 iSeeBetterTrain.py --upscale_factor 2 --RBPN_only --other_dataset True --data_dir afm_dataset_per_sequence_align --file_list sep_trainlist.txt --patch_size 64 --batchSize 12 --denoise --save_folder weights/pretrained2x_mse_Pflow_alignment/ --gpus 2 --useDataParallel --pretrained --pretrained_sr RBPN_2x.pth --use_wandb --optical_flow p

# データセット2 --data_dir ext_clean_dataset_per_sequence
python3 iSeeBetterTrain.py --upscale_factor 2 --RBPN_only --other_dataset True --data_dir ext_clean_dataset_per_sequence --file_list sep_trainlist.txt --patch_size 64 --batchSize 12 --denoise --save_folder weights/pretrained2x_mse_Pflow_blur3/ --gpus 2 --useDataParallel --pretrained --pretrained_sr RBPN_2x.pth --use_wandb --optical_flow p

# パッチサイズ　--patch_size 32 or 64 or 96

# 残差学習 --residual
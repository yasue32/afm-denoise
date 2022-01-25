#train
#2x
python3 iSeeBetterTrain.py --upscale_factor 2 --RBPN_only --other_dataset True --data_dir ./../../../../../home-local/yasue/dataset/afm_dataset4 --file_list sep_trainlist.txt --patch_size 0 --batchSize 16 --denoise --save_folder weights/pretrained2x_mse/ --gpu_id 0 --gpus 1 --pretrained --use_wandb --shuffle --optical_flow p
python3 iSeeBetterTrain.py --upscale_factor 2 --RBPN_only --other_dataset True --data_dir afm_dataset4 --file_list sep_trainlist.txt --patch_size 0 --batchSize 16 --denoise --save_folder weights/pretrained2x_mse/ --gpu_id 0 --gpus 1 --pretrained --use_wandb --shuffle --optical_flow p
#1x RBPN only
python3 iSeeBetterTrain.py --upscale_factor 1 --other_dataset True --data_dir afm_dataset4 --file_list sep_trainlist.txt --patch_size 0 --batchSize 12 --useDataParallel --save_folder weights/ --gpu_id 4 --gpus 1 --denoise --shuffle --RBPN_only --pretrained --pretrained_sr netG_epoch_1_60.pth --alignment
#1x depth
python3 iSeeBetterTrain.py --upscale_factor 1 --other_dataset True --data_dir afm_dataset_depth4 --file_list sep_trainlist.txt --patch_size 0 --batchSize 16 --useDataParallel --save_folder weights/scratch1x_patch4_depth/ --gpu_id 8,9 --gpus 2 --denoise --shuffle --RBPN_only --depth_img


#test
python3 iSeeBetterTest.py --upscale_factor 1 --other_dataset True --data_dir afm_dataset4/211109_2 --file_list sep_trainlist.txt --gpu_id 3 --gpus 1 --model weights/netG_epoch_1_35.pth --gpu_mode --upscale_only --denoise --output Results/scratch1x_patch4_align/
python3 iSeeBetterTest.py --upscale_factor 1 --other_dataset True --data_dir afm_dataset4/211109_2 --file_list sep_trainlist.txt --gpu_mode --gpu_id 3 --gpus 1 --model weights/scratch1x_patch4_align_Ploss/netG_epoch_1_41.pth --upscale_only --denoise --output Results/scratch1x_patch4_align_Ploss/ --alignment
python3 iSeeBetterTest.py --upscale_factor 2 --other_dataset True --data_dir test_dataset_per_sequence --file_list sep_trainlist.txt --gpu_mode --gpu_id 0 --gpus 1 --model weights/pretrained2x_mae_Pflow_blur3/netG_epoch_2_1500.pth --upscale_only --denoise --output Results/pretrained2x_mae_Pflow_blur3/ --optical_flow p

# GPUの指定
CUDA_VISIBLE_DEVICES=x


# akita
python3 iSeeBetterTrain.py --upscale_factor 2 --RBPN_only --other_dataset True --data_dir afm_dataset_per_sequence--file_list sep_trainlist.txt --patch_size 0 --batchSize 16 --denoise --save_folder weights/pretrained2x_mse/ --gpu_id 0 --gpus 1 --pretrained --use_wandb --shuffle --optical_flow p
python3 iSeeBetterTrain.py --upscale_factor 2 --APITLoss --other_dataset True --data_dir ./../../../../../home-local/yasue/dataset/ext_clean_dataset_per_sequence --file_list sep_trainlist.txt --patch_size 0 --batchSize 12 --denoise --save_folder weights/pretrained2x_mse_Pflow_blur3_Aloss015_3/ --gpus 1 --pretrained --pretrained_sr netG_epoch_2_2400.pth --nFrames 7 --nEpochs 10000 --snapshots 150 --pretrained_d netD_epoch_2_2400.pth --start_epoch 2401 --use_wandb
#train
#2x
python3 iSeeBetterTrain.py --upscale_factor 2 --RBPN_only --other_dataset True --data_dir afm_dataset4 --file_list sep_trainlist.txt --patch_size 0 --gpu_id 0 --gpus 1 --denoise --pretrained --pretrained_sr netG_epoch_2_24.pth --use_wandb --shuffle --warping
#1x APITLoss
python3 iSeeBetterTrain.py --upscale_factor 1 --other_dataset True --data_dir afm_dataset4 --file_list sep_trainlist.txt --patch_size 0 --batchSize 25 --useDataParallel --save_folder weights/ --gpu_id 1 --gpus 1 --denoise --shuffle --APITLoss --pretrained --pretrained_sr netG_epoch_1_60.pth --use_wandb --alignment
#1x depth
python3 iSeeBetterTrain.py --upscale_factor 1 --other_dataset True --data_dir afm_dataset_depth4 --file_list sep_trainlist.txt --patch_size 0 --batchSize 15 --useDataParallel --save_folder weights/ --gpu_id 2,3 --gpus 2 --denoise --shuffle --RBPN_only --use_wandb --alignment --depth_img


#test
python3 iSeeBetterTest.py --upscale_factor 1 --other_dataset True --data_dir afm_dataset4/211109_2 --file_list sep_trainlist.txt --gpu_id 3 --gpus 1 --model weights/netG_epoch_1_35.pth --gpu_mode --upscale_only --denoise --output Results/scratch1x_patch4_align/
python3 iSeeBetterTest.py --upscale_factor 1 --other_dataset True --data_dir afm_dataset4/211109_2 --file_list sep_trainlist.txt --gpu_id 3 --gpus 1 --model weights/netG_epoch_1_35.pth --gpu_mode --upscale_only --denoise --output Results/scratch1x_patch4_align_Aloss/ --alignment
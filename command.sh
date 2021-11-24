#train
python3 iSeeBetterTrain.py --upscale_factor 2 --RBPN_only --other_dataset True --data_dir afm_dataset --file_list sep_trainlist.txt --patch_size 0 --gpu_id 0 --denoise --pretrained --pretrained_sr weights/netG_epoch_2_24.pth 
python3 iSeeBetterTrain.py --upscale_factor 2 --RBPN_only --other_dataset True --data_dir afm_dataset --file_list sep_trainlist.txt --patch_size 0 --gpus 3 --gpu_id 4,5,6 --batchSize 3 --useDataParallel --denoise --pretrained --pretrained_sr netG_epoch_2_1.pth

#test
python3 iSeeBetterTest.py --model weights/netG_epoch_4_137.pth --upscale_factor 2 --other_dataset True --data_dir afm_dataset/211029 --file_list sep_trainlist.txt --gpu_id 3

python3 iSeeBetterTest.py --upscale_factor 1 --other_dataset True --data_dir afm_dataset4/211109_2 --file_list sep_trainlist.txt --gpu_id 3 --gpus 1 --model netG_epoch_1_44.pth --gpu_mode --upscale_only --denoise --output Results/scratch1x_patch4_warping/
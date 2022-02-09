# # 素
# python3 iSeeBetterTest.py --upscale_factor 2 --other_dataset True --data_dir dirty_dataset4/ --file_list sep_trainlist.txt --gpu_mode --gpu_id 1 --gpus 1 --model weights/pretrained2x_mse_Pflow/netG_epoch_2_149.pth --upscale_only --denoise --output Results/pretrained2x_mse_Pflow/ --optical_flow p
#python3 iSeeBetterTest.py --upscale_factor 2 --other_dataset True --data_dir /home-local/yasue/dataset/dirty_dataset4/ --file_list sep_trainlist.txt --gpu_mode --gpu_id 1 --gpus 1 --model weights/pretrained2x_mse_Nflow/netG_epoch_2_87.pth --upscale_only --denoise --output Results/pretrained2x_mse_Nflow/ --optical_flow p
#python3 iSeeBetterTest.py --upscale_factor 2 --other_dataset True --data_dir /home-local/yasue/dataset/dirty_dataset4/ --file_list sep_trainlist.txt --gpu_mode --gpu_id 1 --gpus 1 --model weights/pretrained2x_mse_Pflow_blur3/netG_epoch_2_4600.pth --upscale_only --denoise --output Results/pretrained2x_mse_Pflow_blur3/ --optical_flow p
# # + res
#python3 iSeeBetterTest.py --upscale_factor 2 --other_dataset True --data_dir /home-local/yasue/dataset/dirty_dataset4/ --file_list sep_trainlist.txt --gpu_mode --gpu_id 1 --gpus 1 --model weights/pretrained2x_mse_Pflow_res/netG_epoch_2_148.pth --upscale_only --denoise --output Results/pretrained2x_mse_Pflow_res/ --optical_flow p -r
#python3 iSeeBetterTest.py --upscale_factor 2 --other_dataset True --data_dir /home-local/yasue/dataset/dirty_dataset4/ --file_list sep_trainlist.txt --gpu_mode --gpu_id 1 --gpus 1 --model weights/pretrained2x_mse_Pflow_res_blur3/netG_epoch_2_2600.pth --upscale_only --denoise --output Results/pretrained2x_mse_Pflow_res_blur3/ --optical_flow p
# # + Aloss
python3 iSeeBetterTest.py --upscale_factor 2 --other_dataset True --data_dir afm_dataset4/20211109_2 --file_list sep_trainlist.txt --gpu_mode --gpu_id 0 --gpus 1 --model weights/pretrained2x_mse_Pflow_blur3_Aloss015_3/netG_epoch_2_1650.pth --upscale_only --denoise --output Results/pretrained2x_mse_Pflow_blur3_Aloss015_3/ --optical_flow p
python3 iSeeBetterTest.py --upscale_factor 2 --other_dataset True --data_dir afm_dataset4/20211109_2 --file_list sep_trainlist.txt --gpu_mode --gpu_id 0 --gpus 1 --model weights/pretrained2x_mse_Pflow_blur3_Aloss025_3/netG_epoch_2_1800.pth --upscale_only --denoise --output Results/pretrained2x_mse_Pflow_blur3_Aloss025_3/ --optical_flow p
python3 iSeeBetterTest.py --upscale_factor 2 --other_dataset True --data_dir afm_dataset4/20211109_2 --file_list sep_trainlist.txt --gpu_mode --gpu_id 0 --gpus 1 --model weights/pretrained2x_mse_Pflow_blur3_Aloss035_3/netG_epoch_2_1500.pth --upscale_only --denoise --output Results/pretrained2x_mse_Pflow_blur3_Aloss035_3/ --optical_flow p
#python3 iSeeBetterTest.py --upscale_factor 2 --other_dataset True --data_dir afm_dataset4/20211109_2 --file_list sep_trainlist.txt --gpu_mode --gpu_id 1 --gpus 1 --model weights/pretrained2x_mse_Pflow_blur3_Aloss045/netG_epoch_2_3300.pth --upscale_only --denoise --output Results/pretrained2x_mse_Pflow_blur3_Aloss045/ --optical_flow p

#python3 iSeeBetterTest.py --upscale_factor 2 --other_dataset True --data_dir /home-local/yasue/dataset/dirty_dataset4/ --file_list sep_trainlist.txt --gpu_mode --gpu_id 1 --gpus 1 --model weights/pretrained2x_mse_Pflow_Aloss015/netG_epoch_2_1400.pth --upscale_only --denoise --output Results/pretrained2x_mse_Pflow_Aloss015/ --optical_flow p
#python3 iSeeBetterTest.py --upscale_factor 2 --other_dataset True --data_dir /home-local/yasue/dataset/dirty_dataset4/ --file_list sep_trainlist.txt --gpu_mode --gpu_id 1 --gpus 1 --model weights/pretrained2x_mse_Pflow_blur3_Aloss035/netG_epoch_2_1900.pth --upscale_only --denoise --output Results/pretrained2x_mse_Pflow_blur3_Aloss035/ --optical_flow p
#python3 iSeeBetterTest.py --upscale_factor 2 --other_dataset True --data_dir /home-local/yasue/dataset/dirty_dataset4/ --file_list sep_trainlist.txt --gpu_mode --gpu_id 1 --gpus 1 --model weights/pretrained2x_mse_Pflow_Aloss035/netG_epoch_2_900.pth --upscale_only --denoise --output Results/pretrained2x_mse_Pflow_Aloss035/ --optical_flow p
#python3 iSeeBetterTest.py --upscale_factor 2 --other_dataset True --data_dir /home-local/yasue/dataset/dirty_dataset4/ --file_list sep_trainlist.txt --gpu_mode --gpu_id 1 --gpus 1 --model weights/pretrained2x_mse_Pflow_blur3_Aloss035_2/netG_epoch_2_1400.pth --upscale_only --denoise --output Results/pretrained2x_mse_Pflow_blur3_Aloss035_2/ --optical_flow p
export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 0.85 --num_epochs 3   --batch_size 256  --mode train --dataset SMAP  --data_path SMAP --input_c 25    --output_c 25  --loss_fuc MSE --patch_size 357 --win_size 105
python main.py --anormly_ratio 0.85  --num_epochs 10   --batch_size 256     --mode test    --dataset SMAP   --data_path SMAP  --input_c 25    --output_c 25   --loss_fuc MSE --patch_size 357 --win_size 105

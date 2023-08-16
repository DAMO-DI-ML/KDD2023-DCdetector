export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 1 --num_epochs 3   --batch_size 256  --mode train --dataset NIPS_TS_Water  --data_path NIPS_TS_Water  --input_c 9 --output_c 9  --loss_fuc MSE   --patch_size 135  --win_size 90
python main.py --anormly_ratio 1  --num_epochs 10     --batch_size 256   --mode test    --dataset NIPS_TS_Water   --data_path NIPS_TS_Water --input_c 9    --output_c 9    --loss_fuc MSE   --patch_size 135   --win_size 90



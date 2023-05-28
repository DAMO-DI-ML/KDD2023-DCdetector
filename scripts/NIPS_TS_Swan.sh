export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 2.5 --num_epochs 3   --batch_size 256  --mode train --dataset NIPS_TS_Swan  --data_path NIPS_TS_Swan  --input_c 38 --output_c 38  --loss_fuc MSE    --win_size 39
python main.py --anormly_ratio 2.5  --num_epochs 10     --batch_size 256   --mode test    --dataset NIPS_TS_Swan   --data_path NIPS_TS_Swan --input_c 38    --output_c 38    --loss_fuc MSE       --win_size 39

python main.py --anormly_ratio 2 --num_epochs 3   --batch_size 256  --mode train --dataset NIPS_TS_Swan  --data_path NIPS_TS_Swan  --input_c 38 --output_c 38  --loss_fuc MSE    --win_size 39
python main.py --anormly_ratio 2  --num_epochs 10     --batch_size 256   --mode test    --dataset NIPS_TS_Swan   --data_path NIPS_TS_Swan --input_c 38    --output_c 38    --loss_fuc MSE       --win_size 39

python main.py --anormly_ratio 1.8 --num_epochs 3   --batch_size 256  --mode train --dataset NIPS_TS_Swan  --data_path NIPS_TS_Swan  --input_c 38 --output_c 38  --loss_fuc MSE    --win_size 39
python main.py --anormly_ratio 1.8  --num_epochs 10     --batch_size 256   --mode test    --dataset NIPS_TS_Swan   --data_path NIPS_TS_Swan --input_c 38    --output_c 38    --loss_fuc MSE       --win_size 39


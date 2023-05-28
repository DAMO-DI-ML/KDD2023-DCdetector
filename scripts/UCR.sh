export CUDA_VISIBLE_DEVICES=0

for i in {1..250};
do

python main.py --anormly_ratio 0.5 --num_epochs 3   --batch_size 128  --mode train --dataset UCR  --data_path UCR   --input_c 1 --output 1 --index $i --win_size 60 --patch_size 35
python main.py --anormly_ratio 0.5 --num_epochs 10   --batch_size 128    --mode test    --dataset UCR   --data_path UCR     --input_c 1   --output 1  --index $i --win_size 60 --patch_size 35

done  


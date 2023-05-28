export CUDA_VISIBLE_DEVICES=0

# #MSL 
for i in {30,45,60,75,90,105,120,135,150,175,195,210};
do
    python main.py --anormly_ratio 1 --num_epochs 3   --batch_size 128  --mode train --dataset MSL  --data_path MSL  --input_c 55 --output_c 55 --win_size $i --patch_size 35
    python main.py --anormly_ratio 1 --num_epochs 10  --batch_size 128  --mode test  --dataset MSL  --data_path MSL  --input_c 55 --output_c 55 --win_size $i --patch_size 35
done


#SMAP
for i in {30,45,60,75,90,105,120,135,150,175,195,210};
do
    python main.py --anormly_ratio 0.85 --num_epochs 3   --batch_size 128  --mode train --dataset SMAP  --data_path SMAP  --input_c 25 --output_c 25 --win_size $i --patch_size 35
    python main.py --anormly_ratio 0.85 --num_epochs 10  --batch_size 128  --mode test  --dataset SMAP  --data_path SMAP  --input_c 25 --output_c 25 --win_size $i --patch_size 35
done


#PSM
for i in {30,45,60,75,90,105,120,135,150,175,195,210};
do
    python main.py --anormly_ratio 1 --num_epochs 3   --batch_size 128  --mode train --dataset PSM  --data_path PSM  --input_c 25 --output_c 25 --win_size $i --patch_size 35
    python main.py --anormly_ratio 1 --num_epochs 10  --batch_size 128  --mode test  --dataset PSM  --data_path PSM  --input_c 25 --output_c 25 --win_size $i --patch_size 35 
done


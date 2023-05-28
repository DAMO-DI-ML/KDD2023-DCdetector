export CUDA_VISIBLE_DEVICES=0

#MSL 
for j in {1,3,5,13,15,35,135};
do
    python main.py --anormly_ratio 1 --num_epochs 3   --batch_size 128  --mode train --dataset MSL  --data_path MSL  --input_c 55 --output_c 55 --win_size 60 --patch_size $j
    python main.py --anormly_ratio 1 --num_epochs 10  --batch_size 128  --mode test  --dataset MSL  --data_path MSL  --input_c 55 --output_c 55 --win_size 60 --patch_size $j 
done


#PSM
for j in {1,3,5,13,15,35,135};
do
    python main.py --anormly_ratio 1 --num_epochs 5   --batch_size 128  --mode train --dataset PSM  --data_path PSM  --input_c 25 --output_c 25 --win_size 60 --patch_size $j
    python main.py --anormly_ratio 1 --num_epochs 10  --batch_size 128  --mode test  --dataset PSM  --data_path PSM  --input_c 25 --output_c 25 --win_size 60 --patch_size $j 
done


# SMAP
for j in {1,3,5,13,15,35,135};
do
    python main.py --anormly_ratio 0.85 --num_epochs 3   --batch_size 128  --mode train --dataset SMAP  --data_path SMAP  --input_c 25 --output_c 25 --win_size 60 --patch_size $j
    python main.py --anormly_ratio 0.85 --num_epochs 10  --batch_size 128  --mode test  --dataset SMAP  --data_path SMAP  --input_c 25 --output_c 25 --win_size 60 --patch_size $j 
done

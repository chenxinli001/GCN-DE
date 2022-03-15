gpu=1

python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset MRI --organ liver --run_order run1 --pretrain --fuse_type 3  --nlc_layer 567
python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset MRI --organ liver --run_order run2 --pretrain --fuse_type 3  --nlc_layer 567
python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset MRI --organ liver --run_order run3 --pretrain --fuse_type 3  --nlc_layer 567

python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset MRI --organ spleen --run_order run1 --pretrain --fuse_type 3  --nlc_layer 567
python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset MRI --organ spleen --run_order run2 --pretrain --fuse_type 3  --nlc_layer 567
python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset MRI --organ spleen --run_order run3 --pretrain --fuse_type 3  --nlc_layer 567

python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset MRI --organ left_kidney --run_order run1 --pretrain --fuse_type 3  --nlc_layer 567 
python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset MRI --organ left_kidney --run_order run2 --pretrain --fuse_type 3  --nlc_layer 567 
python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset MRI --organ left_kidney --run_order run3 --pretrain --fuse_type 3  --nlc_layer 567 

python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset MRI --organ right_kidney --run_order run1 --pretrain --fuse_type 3  --nlc_layer 567 
python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset MRI --organ right_kidney --run_order run2 --pretrain --fuse_type 3  --nlc_layer 567
python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset MRI --organ right_kidney --run_order run3 --pretrain --fuse_type 3  --nlc_layer 567



# python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset MRI --organ liver --run_order run1 --pretrain --fuse_type 3  --nlc_layer 56767 
# python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset MRI --organ liver --run_order run2 --pretrain --fuse_type 3  --nlc_layer 56767 
# python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset MRI --organ liver --run_order run3 --pretrain --fuse_type 3  --nlc_layer 56767 

# python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset MRI --organ spleen --run_order run1 --pretrain --fuse_type 3  --nlc_layer 56767 
# python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset MRI --organ spleen --run_order run2 --pretrain --fuse_type 3  --nlc_layer 56767 
# python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset MRI --organ spleen --run_order run3 --pretrain --fuse_type 3  --nlc_layer 56767 

# python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset MRI --organ left_kidney --run_order run1 --pretrain --fuse_type 3  --nlc_layer 56767  
# python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset MRI --organ left_kidney --run_order run2 --pretrain --fuse_type 3  --nlc_layer 56767 
# python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset MRI --organ left_kidney --run_order run3 --pretrain --fuse_type 3  --nlc_layer 56767 

# python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset MRI --organ right_kidney --run_order run1 --pretrain --fuse_type 3  --nlc_layer 56767 
# python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset MRI --organ right_kidney --run_order run2 --pretrain --fuse_type 3  --nlc_layer 56767 
# python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset MRI --organ right_kidney --run_order run3 --pretrain --fuse_type 3  --nlc_layer 56767 



# python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset CT --organ liver --run_order run1 --pretrain --fuse_type 3  --nlc_layer 123567789 
# python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset CT --organ liver --run_order run2 --pretrain --fuse_type 3  --nlc_layer 123567789 
# python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset CT --organ liver --run_order run3 --pretrain --fuse_type 3  --nlc_layer 123567789 

# python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset CT --organ spleen --run_order run1 --pretrain --fuse_type 3  --nlc_layer 123567789 
# python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset CT --organ spleen --run_order run2 --pretrain --fuse_type 3  --nlc_layer 123567789 
# python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset CT --organ spleen --run_order run3 --pretrain --fuse_type 3  --nlc_layer 123567789 

# python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset CT --organ left_kidney --run_order run1 --pretrain --fuse_type 3  --nlc_layer 123567789 
# python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset CT --organ left_kidney --run_order run2 --pretrain --fuse_type 3  --nlc_layer 123567789 
# python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset CT --organ left_kidney --run_order run3 --pretrain --fuse_type 3  --nlc_layer 123567789 

# python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset CT --organ right_kidney --run_order run1 --pretrain --fuse_type 3  --nlc_layer 123567789 
# python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset CT --organ right_kidney --run_order run2 --pretrain --fuse_type 3  --nlc_layer 123567789 
# python3 crosseval_main_SE_fuse.py --gpu $gpu --dataset CT --organ right_kidney --run_order run3 --pretrain --fuse_type 3  --nlc_layer 123567789 


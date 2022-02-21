gpu=3

python3 crosseval_main_SE_ms1_cons5_nlc23.py --gpu $gpu --dataset MRI --organ liver --run_order run1 --pretrain --nlc_layer 567 --t 0.1
python3 crosseval_main_SE_ms1_cons5_nlc23.py --gpu $gpu --dataset MRI --organ liver --run_order run2 --pretrain --nlc_layer 567 --t 0.1
python3 crosseval_main_SE_ms1_cons5_nlc23.py --gpu $gpu --dataset MRI --organ liver --run_order run3 --pretrain --nlc_layer 567 --t 0.1

python3 crosseval_main_SE_ms1_cons5_nlc23.py --gpu $gpu --dataset MRI --organ spleen --run_order run1 --pretrain --nlc_layer 567 --t 0.1
python3 crosseval_main_SE_ms1_cons5_nlc23.py --gpu $gpu --dataset MRI --organ spleen --run_order run2 --pretrain --nlc_layer 567 --t 0.1
python3 crosseval_main_SE_ms1_cons5_nlc23.py --gpu $gpu --dataset MRI --organ spleen --run_order run3 --pretrain --nlc_layer 567 --t 0.1

python3 crosseval_main_SE_ms1_cons5_nlc23.py --gpu $gpu --dataset MRI --organ left_kidney --run_order run1 --pretrain --nlc_layer 567 --t 0.1
python3 crosseval_main_SE_ms1_cons5_nlc23.py --gpu $gpu --dataset MRI --organ left_kidney --run_order run2 --pretrain --nlc_layer 567 --t 0.1
python3 crosseval_main_SE_ms1_cons5_nlc23.py --gpu $gpu --dataset MRI --organ left_kidney --run_order run3 --pretrain --nlc_layer 567 --t 0.1

python3 crosseval_main_SE_ms1_cons5_nlc23.py --gpu $gpu --dataset MRI --organ right_kidney --run_order run1 --pretrain --nlc_layer 567 --t 0.1
python3 crosseval_main_SE_ms1_cons5_nlc23.py --gpu $gpu --dataset MRI --organ right_kidney --run_order run2 --pretrain --nlc_layer 567 --t 0.1
python3 crosseval_main_SE_ms1_cons5_nlc23.py --gpu $gpu --dataset MRI --organ right_kidney --run_order run3 --pretrain --nlc_layer 567 --t 0.1
 

# python3 crosseval_main_SE_ms1_cons5_nlc23.py --gpu $gpu --dataset CT --organ liver --run_order run1 --pretrain --nlc_layer 567 --t 0.1
# python3 crosseval_main_SE_ms1_cons5_nlc23.py --gpu $gpu --dataset CT --organ liver --run_order run2 --pretrain --nlc_layer 567 --t 0.1
# python3 crosseval_main_SE_ms1_cons5_nlc23.py --gpu $gpu --dataset CT --organ liver --run_order run3 --pretrain --nlc_layer 567 --t 0.1

# python3 crosseval_main_SE_ms1_cons5_nlc23.py --gpu $gpu --dataset CT --organ spleen --run_order run1 --pretrain --nlc_layer 567 --t 0.1
# python3 crosseval_main_SE_ms1_cons5_nlc23.py --gpu $gpu --dataset CT --organ spleen --run_order run2 --pretrain --nlc_layer 567 --t 0.1
# python3 crosseval_main_SE_ms1_cons5_nlc23.py --gpu $gpu --dataset CT --organ spleen --run_order run3 --pretrain --nlc_layer 567 --t 0.1

# python3 crosseval_main_SE_ms1_cons5_nlc23.py --gpu $gpu --dataset CT --organ left_kidney --run_order run1 --pretrain --nlc_layer 567 --t 0.1
# python3 crosseval_main_SE_ms1_cons5_nlc23.py --gpu $gpu --dataset CT --organ left_kidney --run_order run2 --pretrain --nlc_layer 567 --t 0.1
# python3 crosseval_main_SE_ms1_cons5_nlc23.py --gpu $gpu --dataset CT --organ left_kidney --run_order run3 --pretrain --nlc_layer 567 --t 0.1

# python3 crosseval_main_SE_ms1_cons5_nlc23.py --gpu $gpu --dataset CT --organ right_kidney --run_order run1 --pretrain --nlc_layer 567 --t 0.1
# python3 crosseval_main_SE_ms1_cons5_nlc23.py --gpu $gpu --dataset CT --organ right_kidney --run_order run2 --pretrain --nlc_layer 567 --t 0.1
# python3 crosseval_main_SE_ms1_cons5_nlc23.py --gpu $gpu --dataset CT --organ right_kidney --run_order run3 --pretrain --nlc_layer 567 --t 0.1





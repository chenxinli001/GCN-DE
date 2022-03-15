gpu=3

python3 crosseval_main_GC.py --gpu $gpu --dataset MRI --organ liver --run_order run1 --pretrain 
python3 crosseval_main_GC.py --gpu $gpu --dataset MRI --organ liver --run_order run2 --pretrain 
python3 crosseval_main_GC.py --gpu $gpu --dataset MRI --organ liver --run_order run3 --pretrain 

python3 crosseval_main_GC.py --gpu $gpu --dataset MRI --organ spleen --run_order run1 --pretrain 
python3 crosseval_main_GC.py --gpu $gpu --dataset MRI --organ spleen --run_order run2 --pretrain 
python3 crosseval_main_GC.py --gpu $gpu --dataset MRI --organ spleen --run_order run3 --pretrain 

python3 crosseval_main_GC.py --gpu $gpu --dataset MRI --organ left_kidney --run_order run1 --pretrain  
python3 crosseval_main_GC.py --gpu $gpu --dataset MRI --organ left_kidney --run_order run2 --pretrain  
python3 crosseval_main_GC.py --gpu $gpu --dataset MRI --organ left_kidney --run_order run3 --pretrain  

python3 crosseval_main_GC.py --gpu $gpu --dataset MRI --organ right_kidney --run_order run1 --pretrain  
python3 crosseval_main_GC.py --gpu $gpu --dataset MRI --organ right_kidney --run_order run2 --pretrain 
python3 crosseval_main_GC.py --gpu $gpu --dataset MRI --organ right_kidney --run_order run3 --pretrain 

# gpu=0

# python crosseval_main_MIA.py --gpu $gpu --dataset MRI --organ liver --method 'Shaban'
# python crosseval_main_MIA.py --gpu $gpu --dataset CT --organ liver --method 'Shaban'

# python crosseval_main_MIA.py --gpu $gpu --dataset MRI --organ liver --method 'Rakelly'
# python crosseval_main_MIA.py --gpu $gpu --dataset CT --organ liver --method 'Rakelly'


# gpu=1
# python crosseval_main_MIA.py --gpu $gpu --dataset MRI --organ spleen --method 'Shaban'
# python crosseval_main_MIA.py --gpu $gpu --dataset CT --organ spleen --method 'Shaban'

# python crosseval_main_MIA.py --gpu $gpu --dataset MRI --organ spleen --method 'Rakelly'
# python crosseval_main_MIA.py --gpu $gpu --dataset CT --organ spleen --method 'Rakelly'


# gpu=2
# python crosseval_main_MIA.py --gpu $gpu --dataset MRI --organ left_kidney --method 'Shaban'
# python crosseval_main_MIA.py --gpu $gpu --dataset CT --organ left_kidney --method 'Shaban'

# python crosseval_main_MIA.py --gpu $gpu --dataset MRI --organ left_kidney --method 'Rakelly'
# python crosseval_main_MIA.py --gpu $gpu --dataset CT --organ left_kidney --method 'Rakelly'


gpu=3
python crosseval_main_MIA.py --gpu $gpu --dataset MRI --organ right_kidney --method 'Shaban'
python crosseval_main_MIA.py --gpu $gpu --dataset CT --organ right_kidney --method 'Shaban'

python crosseval_main_MIA.py --gpu $gpu --dataset MRI --organ right_kidney --method 'Rakelly'
python crosseval_main_MIA.py --gpu $gpu --dataset CT --organ right_kidney --method 'Rakelly'




# Hierarchical Deep Network with Uncertainty aware Semi-supervised Learning for Vesse Segmentation (CBM 2021) ([Link](https://www.sciencedirect.com/science/article/pii/S0010482521008611))

A Pytorch Implementation of ''Hierarchical Deep Network with Uncertainty aware Semi-supervised Learning for Vesse Segmentation'', which is accepted by the jounal of Computers in Biology and Medicine.

## Requirements

- Python == 3.7.4
- Pytorch == 1.1.0
- Torchvision == 0.3.0
- CUDA 8.0

## Train
python main_ours.py

## Test
python test.py

## Results
### Quantitative results on MRI measured in DC scores 
|  Organ           |  Liver |  Spleen | Left Kidney  |  Right Kidney | mean  | 
|  --------------  |  ----  |  ------ | -----------  |  ------------ | ----  |
|  OSLSM           |  25.73 |  34.66  | 29.21        |  22.61        | 28.00 |
|  co-FCN          |**53.74**|  57.41 | 60.62        |  71.13        | 60.70 |
|  PANet           |  51.37 |  43.59  | 25.54        |  26.45        | 36.74 |
|  SG-One          |  50.33 |  42.41  | 26.79        |  24.16        | 35.92 |
|  SE-FSS          |  40.32 |  48.93  | 62.56        |  65.81        | 54.38 |
|  GCN             |  51.33 |  58.67  | 63.67        |  70.33        | 61.00 |
|  GCN-DE(Ours)    |  49.47 |**60.63**| **76.07**    |  **83.03**    | **67.30**|

### Quantitative results on CT measured in DC scores 
|  Organ           |  Liver |  Spleen | Left Kidney  |  Right Kidney | mean  | 
|  --------------  |  ----  |  ------ | -----------  |  ------------ | ----  |
|  OSLSM           |  29.65 |  19.40  | 15.82        |  7.54         | 18.08 |
|  co-FCN          |**47.50**|  43.86 | 41.30        |  33.51        | 41.53 |
|  PANet           |  44.25 |  30.49  | 25.30        |  22.95        | 30.75 |
|  SG-One          |  44.98 |  30.88  | 26.79        |  20.88        | 30.88 |
|  SE-FSS          |  44.51 |  40.52  | 40.10        |  34.80        | 39.97 |
|  GCN             |  47.00 |  46.67  | 42.33        |  35.00        | 42.75 |
|  GCN-DE(Ours)    |  46.77 |**56.53**| **68.13**    |  **75.50**    | **61.73**|

## Citation

If you find this repository useful, please cite our paper:
```
@article{sun2022few,
  title={Few-shot medical image segmentation using a global correlation network with discriminative embedding},
  author={Sun, Liyan and Li, Chenxin and Ding, Xinghao and Huang, Yue and Chen, Zhong and Wang, Guisheng and Yu, Yizhou and Paisley, John},
  journal={Computers in biology and medicine},
  volume={140},
  pages={105067},
  year={2022},
  publisher={Elsevier}
}
```


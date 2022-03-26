# FSL-using-GCN-DE
## Requirements

* Python == 
* Pytorch == 
* Torchvision ==

## Train
python main_ours.py

## Test
python test.py

## Quantitative results measured in DC scores on MRI
|  Organ           |  Liver |  Spleen | Left Kidney  |  Right Kidney | mean  | 
|  --------------  |  ----  |  ------ | -----------  |  ------------ | ----  |
|  OSLSM           |  25.73 |  34.66  | 29.21        |  22.61        | 28.00 |
|  co-FCN          |  53.74 |  57.41  | 60.62        |  71.13        | 60.70 |
|  PANet           |  51.37 |  43.59  | 25.54        |  26.45        | 36.74 |
|  SG-One          |  50.33 |  42.41  | 26.79        |  24.16        | 35.92 |
|  SE-FSS          |  40.32 |  48.93  | 62.56        |  65.81        | 54.38 |
|  GCN             |  51.33 |  58.67  | 63.67        |  70.33        | 61.00 |
|  GCN-DE(Ours)    |  49.47 |  60.63  | 76.07        |  83.03        | **67.30** |

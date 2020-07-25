# Show, Attend and Tell: Pytorch Implementation
## Objective
This repo is a Image Captioning Model, which is in the **"Show, Attend and Tell: Neural Image Caption Generation with Visual Attention (ICML'15)"** paper.
## Results
| BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
| ------ | ------ | ------ | ------ |
| 61.09  | 36.54  | 22.64  | 14.21  |

![image](https://user-images.githubusercontent.com/37788686/88451050-bf658200-ce8e-11ea-8875-5d6f5a46b104.png)
## Used In
**DATASET:** Flickr8k  
**ENCODER LAYER:** pretrained Vgg16Net

## To train
`Python3 main.py`

## To test
`Python3 main.py --test`

## Reference
[Show Attend and Tell](https://arxiv.org/pdf/1502.03044.pdf)

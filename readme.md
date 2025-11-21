# Unsupervised Multi-semantic Similarity Reconstruction Contrastive Hashing for Multi-label Image Retrieval

This is the implementation of paper: Unsupervised Multi-semantic Similarity Reconstruction Contrastive Hashing for Multi-label Image Retrieval.(DOI: [](10.1016/j.eswa.2025.130414))

## Datasets 

Experiments on **3 image datasets**: [FLICKR25K](https://press.liacs.nl/mirflickr/mirdownload.html), [COCO2014](https://github.com/thuml/HashNet/blob/master/pytorch/README.md), [NUSWIDE](https://github.com/thuml/HashNet/blob/master/pytorch/README.md).

## Dependencies

- Python 3.8
- Pytorch 1.10.1
- torchvision 0.11.2
- CUDA 11.1

## Training Example

```shell
python train.py with umrch flickr hash_bit=16
```

## Model Weights
Download: [Link](https://pan.baidu.com/s/1cfmYJ2a0xadsuolVU9FDEw?pwd=mwcx) 

## Test Example

```shell
python metric.py --dataset flickr --hash_bit 16 --method umrch --iscode --backbone_frozen --ckpt checkpoints/umrch_logs/best_umrch_100_flickr_clip_True_16_32/*_best.pth
```

# Unsupervised Multi-semantic Similarity Reconstruction Contrastive Hashing for Multi-label Image Retrieval

This is the implementation of paper: Unsupervised Multi-semantic Similarity Reconstruction Contrastive Hashing for Multi-label Image Retrieval.

## Datasets 

Experiments on **3 image datasets**: FLICKR25K, COCO2014, NUSWIDE。

Download：[Link](https://github.com/thuml/HashNet/blob/master/pytorch/README.md) 

### Dependencies

- Python 3.8
- Pytorch 1.10.1
- torchvision 0.11.2
- CUDA 11.1

### Training example

```shell
python train.py with umrch flickr hash_bit=16
```
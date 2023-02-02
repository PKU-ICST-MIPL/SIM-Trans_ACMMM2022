# Introduction

This is the source code of our ACM MM 2022 paper "SIM-Trans: Structure Information Modeling Transformer for Fine-grained Visual Categorization". Please cite the following paper if you use our code.

Hongbo Sun, Xiangteng He and Yuxin Peng, "SIM-Trans: Structure Information Modeling Transformer for Fine-grained Visual Categorization", 30th ACM Multimedia Conference (ACM MM), 2022.




# Dependencies

Python 3.7.7

PyTorch 1.5.0

Torchvision 0.6.0



# Data Preparation

Download the CUB-200-2011 dataset and iNaturalist 2017 dataset from official websites and put them in corresponding folders.



# Usage

Start training by executing the following commands. This will train the model on CUB-200-2011 dataset.

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 10715 --nproc_per_node=4 train.py --dataset CUB_200_2011 --split overlap --num_steps 10000  --eval_every 1000 --fp16 --name sample_run --train_batch_size 5

For any questions, feel free to contact us (sunhongbo@pku.edu.cn).

Welcome to our [Laboratory Homepage](http://www.icst.pku.edu.cn/mipl/home/) for more information about our papers, source codes, and datasets.


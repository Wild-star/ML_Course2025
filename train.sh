#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=18
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00

cd /public/home/tangzili/stereo/machinelearn
conda activate monster
python train.py --epochs 10000 --batch_size 128 --hidden_size 64 --n_runs 3 --seq_len 90 \
--pred_len 90 

python train_pro.py --epochs 10000 --batch_size 128 --hidden_size 64 --n_runs 3 --seq_len 90 \
--pred_len 90 

python train.py --epochs 10000 --batch_size 128 --hidden_size 64 --n_runs 3 --seq_len 365 \
--pred_len 365

python train_pro.py --epochs 10000 --batch_size 128 --hidden_size 64 --n_runs 3 --seq_len 365 \
--pred_len 365

python train_tran.py --epochs 1500 --batch_size 256 --lr 0.0001 --seq_len 90 --pred_len 90

python train_tran.py --epochs 1500 --batch_size 256 --lr 0.0001 --seq_len 90 --pred_len 365

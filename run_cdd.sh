#!/usr/bin/sh

python main.py --dataset Core50 --split_core --strategy CDD --seed 0  --device 1 --buffer_size 1000 --epochs 1 
python main.py --dataset Core50 --split_core --strategy CDD --seed 1  --device 1 --buffer_size 1000 --epochs 1 
python main.py --dataset Core50 --split_core --strategy CDD --seed 2  --device 1 --buffer_size 1000 --epochs 1 
python main.py --dataset Core50 --split_core --strategy CDD --seed 3  --device 1 --buffer_size 1000 --epochs 1 
python main.py --dataset Core50 --split_core --strategy CDD --seed 4  --device 1 --buffer_size 1000 --epochs 1 

python main.py --dataset Core50 --split_core --strategy CDD --seed 0  --device 1 --buffer_size 1000 --epochs 10
python main.py --dataset Core50 --split_core --strategy CDD --seed 1  --device 1 --buffer_size 1000 --epochs 10
python main.py --dataset Core50 --split_core --strategy CDD --seed 2  --device 1 --buffer_size 1000 --epochs 10
python main.py --dataset Core50 --split_core --strategy CDD --seed 3  --device 1 --buffer_size 1000 --epochs 10
python main.py --dataset Core50 --split_core --strategy CDD --seed 4  --device 1 --buffer_size 1000 --epochs 10

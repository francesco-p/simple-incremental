#!/usr/bin/sh

python main.py --dataset Core50 --strategy replay --seed 0  --device 0 --buffer_size 1000 --epochs 1 
python main.py --dataset Core50 --strategy replay --seed 1  --device 0 --buffer_size 1000 --epochs 1 
python main.py --dataset Core50 --strategy replay --seed 2  --device 0 --buffer_size 1000 --epochs 1 
python main.py --dataset Core50 --strategy replay --seed 3  --device 0 --buffer_size 1000 --epochs 1 
python main.py --dataset Core50 --strategy replay --seed 4  --device 0 --buffer_size 1000 --epochs 1 

python main.py --dataset Core50 --strategy replay --seed 0  --device 0 --buffer_size 1000 --epochs 10
python main.py --dataset Core50 --strategy replay --seed 1  --device 0 --buffer_size 1000 --epochs 10
python main.py --dataset Core50 --strategy replay --seed 2  --device 0 --buffer_size 1000 --epochs 10
python main.py --dataset Core50 --strategy replay --seed 3  --device 0 --buffer_size 1000 --epochs 10
python main.py --dataset Core50 --strategy replay --seed 4  --device 0 --buffer_size 1000 --epochs 10

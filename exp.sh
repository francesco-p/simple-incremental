#!/usr/bin/sh


#python main.py --dataset Core50 --strategy CDD --seed 0  > /tmp/cdd.txt &
python main.py --dataset Core50 --strategy finetuning --seed 0 > /tmp/finetuning.txt 
python main.py --dataset Core50 --strategy replay --seed 0 > /tmp/replay.txt 
python main.py --dataset Core50 --strategy finetuningfc --seed 0 > /tmp/replay.txt

#python main.py --num_tasks 20 --strategy finetuning > /tmp/ft.txt &
#python main.py --num_tasks 20 --strategy finetuningfc > /tmp/ftfc.txt &
#python main.py --num_tasks 20 --strategy surgicalft --surgical_layer 3 > /tmp/surgical.txt &
#python main.py --num_tasks 20 --strategy replay --buffer_size 500 > /tmp/replay.txt 

#!/usr/bin/sh

# lo dico due volte il device perche sono bravo a programmare
python CDD/split_dset.py --seed 0 --device 0 --gpu_id 0 --num_seed_vec 1 --num_decoder 20
python CDD/split_dset.py --seed 1 --device 0 --gpu_id 0 --num_seed_vec 1 --num_decoder 20
python CDD/split_dset.py --seed 2 --device 0 --gpu_id 0 --num_seed_vec 1 --num_decoder 20
python CDD/split_dset.py --seed 3 --device 0 --gpu_id 0 --num_seed_vec 1 --num_decoder 20
python CDD/split_dset.py --seed 4 --device 0 --gpu_id 0 --num_seed_vec 1 --num_decoder 20


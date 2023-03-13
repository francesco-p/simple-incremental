#!/usr/bin/sh

python CDD/split_dset.py --seed 0 --device 0 --cdd_num_seed_vec 1 --cdd_num_decoder 20
python CDD/split_dset.py --seed 1 --device 0 --cdd_num_seed_vec 1 --cdd_num_decoder 20
python CDD/split_dset.py --seed 2 --device 0 --cdd_num_seed_vec 1 --cdd_num_decoder 20
python CDD/split_dset.py --seed 3 --device 0 --cdd_num_seed_vec 1 --cdd_num_decoder 20
python CDD/split_dset.py --seed 4 --device 0 --cdd_num_seed_vec 1 --cdd_num_decoder 20


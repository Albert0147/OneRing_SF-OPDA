#!/bin/bash


python train_officehome_sfunda.py  --gpu 0  --file logfile  --model_name sfunda --save_model  --dset p2a --lpa
python train_officehome_sfunda.py  --gpu 0  --file logfile  --model_name sfunda --save_model  --dset p2c --lpa
python train_officehome_sfunda.py --gpu 0  --file logfile  --model_name sfunda --save_model  --dset p2r --lpa

python train_officehome_sfunda.py  --gpu 0  --file logfile  --model_name sfunda --save_model  --dset a2c --lpa
python train_officehome_sfunda.py  --gpu 0  --file logfile  --model_name sfunda --save_model  --dset a2p --lpa
python train_officehome_sfunda.py  --gpu 0  --file logfile  --model_name sfunda --save_model  --dset a2r --lpa

python train_officehome_sfunda.py  --gpu 0  --file logfile  --model_name sfunda --save_model  --dset c2a --lpa
python train_officehome_sfunda.py  --gpu 0  --file logfile  --model_name sfunda --save_model  --dset c2p --lpa
python train_officehome_sfunda.py  --gpu 0  --file logfile  --model_name sfunda --save_model  --dset c2r --lpa


python train_officehome_sfunda.py  --gpu 0  --file logfile  --model_name sfunda --save_model  --dset r2a --lpa
python train_officehome_sfunda.py  --gpu 0  --file logfile  --model_name sfunda --save_model  --dset r2c --lpa
python train_officehome_sfunda.py  --gpu 0  --file logfile  --model_name sfunda --save_model  --dset r2p --lpa
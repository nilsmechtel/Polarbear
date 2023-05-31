#!/bin/bash

# train Polarbear model

data_dir=/workspace/data/ArchR/filtered_feature_bc_matrix
cur_dir=.
device=$1

## choose GPU to run model on
export CUDA_VISIBLE_DEVICES=${device}

## train the model
python ${cur_dir}/bin/run_polarbear.py --path_x_y ${data_dir} --outdir ${cur_dir}/output_nb_coassay_gpu/ --patience 45 --train_test_split random --train train

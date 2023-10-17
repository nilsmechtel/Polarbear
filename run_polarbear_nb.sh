#!/bin/bash
# train Polarbear model with NB and PCPG data

cur_dir=.

path_x_list=(
    "/projects/23_NB_Multiome/B087-NB_13363_normal2/filtered_feature_bc_matrix"
    "/projects/23_NB_Multiome/B087-NB_14766_normal2/filtered_feature_bc_matrix"
    "/projects/23_NB_Multiome/B087-NB_18800_tumor2/filtered_feature_bc_matrix"
    "/projects/23_NB_Multiome/B087-NB_20488_tumor1/filtered_feature_bc_matrix"
    "/projects/23_NB_Multiome/B087-NB_21302_tumor1/filtered_feature_bc_matrix"
    "/projects/23_NB_Multiome/B087-NB_21997_tumor1/filtered_feature_bc_matrix"
    "/projects/23_NB_Multiome/B087-NB_22650_tumor2/filtered_feature_bc_matrix"
    "/projects/23_NB_Multiome/B087-NB_22951_tumor1/filtered_feature_bc_matrix"
    "/projects/23_NB_Multiome/B087-NB_23040_tumor1/filtered_feature_bc_matrix"
    "/projects/23_NB_Multiome/B087-NB_26400_tumor1/filtered_feature_bc_matrix"
    "/projects/23_NB_Multiome/B087-NB_27266_tumor1/filtered_feature_bc_matrix"
    "/projects/23_NB_Multiome/OE0415_PCPG_GYDR-0199_tumor1/filtered_feature_bc_matrix"
    "/projects/23_NB_Multiome/OE0415_PCPG_GYDR-0211_tumor1/filtered_feature_bc_matrix"
    "/projects/23_NB_Multiome/OE0415_PCPG_GYDR-0533_tumor1/filtered_feature_bc_matrix"
    "/projects/23_NB_Multiome/OE0415_PCPG_GYDR-0558_tumor1/filtered_feature_bc_matrix"
)
path_y_list=(
    "/projects/23_NB_Multiome/ArchR/B087-NB_13363_normal2"
    "/projects/23_NB_Multiome/ArchR/B087-NB_14766_normal2"
    "/projects/23_NB_Multiome/ArchR/B087-NB_18800_tumor2"
    "/projects/23_NB_Multiome/ArchR/B087-NB_20488_tumor1"
    "/projects/23_NB_Multiome/ArchR/B087-NB_21302_tumor1"
    "/projects/23_NB_Multiome/ArchR/B087-NB_21997_tumor1"
    "/projects/23_NB_Multiome/ArchR/B087-NB_22650_tumor2"
    "/projects/23_NB_Multiome/ArchR/B087-NB_22951_tumor1"
    "/projects/23_NB_Multiome/ArchR/B087-NB_23040_tumor1"
    "/projects/23_NB_Multiome/ArchR/B087-NB_26400_tumor1"
    "/projects/23_NB_Multiome/ArchR/B087-NB_27266_tumor1"
    "/projects/23_NB_Multiome/ArchR/OE0415_PCPG_GYDR-0199_tumor1"
    "/projects/23_NB_Multiome/ArchR/OE0415_PCPG_GYDR-0211_tumor1"
    "/projects/23_NB_Multiome/ArchR/OE0415_PCPG_GYDR-0533_tumor1"
    "/projects/23_NB_Multiome/ArchR/OE0415_PCPG_GYDR-0558_tumor1"
)

path_x=""
for item in "${path_x_list[@]}"; do
    path_x="$path_x $item"
done
path_x="${path_x# }"

path_y=""
for item in "${path_y_list[@]}"; do
    path_y="$path_y $item"
done
path_y="${path_y# }"

## choose GPU to run model on
device=$1
export CUDA_VISIBLE_DEVICES=${device}

## choose gene and peak set
gene_set="/workspace/PELICAN/preprocessing/NB_PCPG/reduced_gene_set.txt"
peak_set="/workspace/PELICAN/preprocessing/NB_PCPG/reduced_peak_set.txt"

## train the model
python ${cur_dir}/bin/run_polarbear.py --path_x $path_x --path_y $path_y --outdir ${cur_dir}/output_nb_only-annotated/ --patience 45 --train_test_split pelican --gene_set $gene_set --peak_set $peak_set --train train --predict predict

## evaluate and output predictions on test set
# python ${cur_dir}/bin/run_polarbear.py --path_x $path_x --path_y $path_y --outdir ${cur_dir}/output_nb_only-annotated/ --patience 45 --train_test_split pelican --gene_set $gene_set --peak_set $peak_set --train predict --predict predict
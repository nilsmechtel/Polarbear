#!/usr/bin/env python

import os, sys, argparse, random
import tensorflow as tf
from tensorflow.python.keras import backend as K
import numpy as np
import pandas as pd
import gzip
from sklearn import preprocessing, metrics
from sklearn.linear_model import RidgeCV
import scipy
from scipy.io import mmread
from copy import copy
import neptune.new as neptune
from datetime import datetime

sys.path.append("bin/")
from train_model import TranslateAE
from evaluation_functions import *

"""
generate_output.py builds separate autoencoders for each domain, and then connects two bottle-neck layers with translator layer. 
Loss = reconstr_loss of each domain + prediction_error of co-assay data from one domain to the other.
"""


def load_rna_file_sparse(url1):
    """
    load scRNA mtx file to sparse matrix, binarize the data; load batch info
    """
    start_time = datetime.now()
    print("Loading RNA data...", end=" ", flush=True)
    # logger.info('url1={}'.format(url1)); assert os.path.isfile(url1);
    data_rna = mmread(url1).tocsr()
    data_rna = data_rna.astype(int)
    elapsed_time = datetime.now() - start_time
    print(f"done! (elapsed time: {str(elapsed_time).split('.')[0]})", end="\n", flush=True)
    data_rna_batch = mmread(url1.split(".mtx")[0] + "_barcodes_dataset.mtx").tocsr()
    return data_rna, data_rna_batch


def load_atac_file_sparse(url2, index_range):
    """
    load scATAC mtx file to sparse matrix, binarize the data; load batch info
    index_range: specify the index of peaks to be selected - e.g. filter out non-auto chromosome peaks
    """
    start_time = datetime.now()
    print("Loading ATAC data...", end=" ", flush=True)
    # logger.info('url2={}'.format(url2)); assert os.path.isfile(url2);
    data_atac = mmread(url2).tocsr()[:, index_range]
    data_atac[data_atac != 0] = 1
    data_atac = data_atac.astype(int)
    elapsed_time = datetime.now() - start_time
    print(f"done! (elapsed time: {str(elapsed_time).split('.')[0]})", end="\n", flush=True)
    data_atac_batch = mmread(url2.split(".mtx")[0] + "_barcodes_dataset.mtx").tocsr()
    return data_atac, data_atac_batch


def find_samples_from_directory(path_list):
    sample_folders = [p.strip("/").split("/") for p in path_list]
    flattened_folders = [folder for folders in sample_folders for folder in folders]
    unique_folders = np.unique(np.array(flattened_folders))
    folder_counts = np.array([
        sum([folder in folders for folders in sample_folders])
        for folder in unique_folders
    ])
    possible_samples = set(unique_folders[
        np.logical_and(
            folder_counts != 0,
            folder_counts != len(path_list)
        )
    ])
    return possible_samples

def read_gzip_matrixmarket(file_path):
    with gzip.open(file_path, "rt") as f:
        MatrixMarket, matrix, coordinate, integer_real, general = (
            f.readline().strip().split(" ")
        )
        dtype = int if integer_real == "integer" else float
        comment = True
        while comment is True:
            line = f.readline().strip()
            comment = line.startswith("%")
        rows, cols, entries = map(int, line.split(" "))
        # Read matrix data into pandas dataframe
        matrix_df = pd.read_csv(
            f, sep=" ", header=None, names=["row", "col", "value"], dtype=dtype
        )
        # Extract matrix_data
        row_data = matrix_df["row"].values - 1  # zero based indexing
        col_data = matrix_df["col"].values - 1  # zero based indexing
        value_data = matrix_df["value"].values

        matrix_sparse = scipy.sparse.csr_matrix(
            (value_data, (row_data, col_data)), shape=(rows, cols), dtype=dtype
        )
        return matrix_sparse


def load_10x_data_sparse(directory, sample, modality, supported_features):
    files = os.listdir(directory)

    # Matrix file
    matrix_f = [
        os.path.join(directory, file) for file in files if file.startswith("matrix")
    ]
    if len(matrix_f) != 1:
        raise ValueError(
            f"{directory}: Expected one matrix file but got {len(matrix_f)}"
        )
    else:
        matrix_f = matrix_f[0]

    # Barcode file
    barcode_f = [
        os.path.join(directory, file) for file in files if file.startswith("barcodes")
    ]
    if len(barcode_f) != 1:
        raise ValueError(
            f"{directory}: Expected one barcode file but got {len(barcode_f)}"
        )
    else:
        barcode_f = barcode_f[0]

    # Features file
    features_f = [
        os.path.join(directory, file) for file in files if file.startswith("features")
    ]
    if len(features_f) != 1:
        raise ValueError(
            f"{directory}: Expected one feature file but got {len(features_f)}"
        )
    else:
        features_f = features_f[0]

    # Load barcodes
    with gzip.open(barcode_f, "rt") as file:
        barcodes = [barcode.strip() for barcode in file.readlines()]

    # Add sample to each barcode (if not already done) to keep barcodes unique across samples
    if "#" not in barcodes[0]:
        barcodes = [f"{sample}#{barcode}" for barcode in barcodes]

    # Load features
    features_df = pd.read_csv(
        features_f,
        delimiter="\t",
        compression="gzip" if features_f.endswith(".gz") else None,
        header=None,
    )
    if features_df.shape[1] == 6:
        features_df.columns = [
            "feature_id",
            "feature_name",
            "feature_type",
            "chromosome",
            "start",
            "end",
        ]
    elif features_df.shape[1] == 3:
        features_df.columns = [
            "chromosome",
            "start",
            "end",
        ]
        features_df["feature_name"] = (
            features_df["chromosome"]
            + ":"
            + features_df["start"].astype(str)
            + "-"
            + features_df["end"].astype(str)
        )
        features_df["feature_type"] = "Peaks"
    else:
        raise RuntimeError(
            f"Unknown feature input. Expected 3 or 6 columns but got {features_df.shape[1]}"
        )    

    # Filter out unwanted features
    for colname, supported_features in supported_features.items():
        features_df = features_df.loc[
            features_df[colname].isin(set(supported_features)), :
        ]

    # Sort by 'chromosome', 'start' and 'end' of region
    chromosome_order = ["chr" + str(i) for i in list(range(1, 23)) + ["X", "Y"]]
    features_df["chromosome"] = pd.Categorical(
        features_df["chromosome"], categories=chromosome_order, ordered=True
    ).remove_unused_categories()
    features_df["start"] = pd.to_numeric(features_df["start"])
    features_df["end"] = pd.to_numeric(features_df["end"])
    features_df = features_df.sort_values(by=["chromosome", "start", "end"])
    

    # Load matrix
    matrix = read_gzip_matrixmarket(matrix_f).astype(float)

    # Get feature indices of chosen modality
    if modality == "RNA":
        features_df = features_df.loc[features_df.feature_type == "Gene Expression", :]
        feature_index = features_df.index.to_list()
        matrix = matrix[feature_index, :]

    else:
        features_df = features_df.loc[features_df.feature_type == "Peaks", :]
        feature_index = features_df.index.to_list()
        matrix = matrix[feature_index, :]

        # Binarize ATAC peak matrix
        matrix[matrix > 1] = 1

    # check if matrix needs to be transposed:
    # (feature * cells) -> (cells * features)
    transpose = matrix.shape[1] == len(barcodes)
    if transpose is True:
        matrix = matrix.transpose()

    return matrix.tocsr(), barcodes, features_df


def concatenate_samples(matrix_list, batch_list, feature_list, barcode_list):
    # Check if all loaded features are the same
    all_same = all(feature_list[0].equals(features) for features in feature_list)  # pd.DataFrame
    if not all_same:
        raise ValueError("Not all features are the same!")

    # Concatenate all each list of data
    data = scipy.sparse.vstack(matrix_list)
    data_batch = scipy.sparse.vstack(batch_list)
    features = feature_list[0]

    # Check for same number of cells in genomic and batch data
    if not data.shape[0] == data_batch.shape[0] == len(barcode_list):
        raise RuntimeError(
            f"Expected {len(barcode_list)} barcodes in count matrix but got {data.shape[0]}."
        )
    # Check for same number of features
    if not data.shape[1] == len(features):
        raise RuntimeError(
            f"Expected {len(features)} features in count matrix but got {data.shape[1]}."
        )

    return data.tocsr(), data_batch.tocsr(), features


# def load_10x_barcodes(path_x_y):
#     """
#     read cell barcodes into a list and one-hot encode batch factors

#     format: (ncell * batch_dim)
#     """
#     start_time = datetime.now()
#     print("Loading cell barcodes...", end=" ", flush=True)
#     ## cell barcodes
#     barcodes = pd.read_csv(
#         os.path.join(path_x_y, "barcodes.tsv.gz"),
#         header=None,
#         delimiter="\t",
#         compression="gzip",
#     )
#     barcode_list = barcodes[0].to_list()
#     ## encode batch factor
#     ncell = len(barcode_list)

#     if all(["#" in id for id in barcode_list]):
#         samples = np.array([id.split("#")[0] for id in barcode_list])
#     else:
#         samples = np.array(["sample1" for id in barcode_list])

#     batch_dim = len(np.unique(samples))
#     data_batch = np.zeros(
#         shape=(ncell, batch_dim),
#         dtype=int,
#     )
#     for batch_factor, sample in enumerate(np.unique(samples)):
#         data_batch[np.where(samples == sample), batch_factor] = 1
#     data_batch = scipy.sparse.csr_matrix(data_batch)

#     elapsed_time = datetime.now() - start_time
#     print(f"done! (elapsed time: {str(elapsed_time).split('.')[0]})", end="\n", flush=True)
#     return barcode_list, data_batch


# def load_10x_features(path_x_y):
#     """
#     load features into pandas DataFrame and filter out genes & peaks in sex chromosomes
#     format: (features * col_names)
#     """
#     start_time = datetime.now()
#     print("Loading RNA and ATAC features...", end="\t", flush=True)
#     # load features
#     features = pd.read_csv(
#         os.path.join(path_x_y, "features.tsv.gz"),
#         delimiter="\t",
#         compression="gzip",
#         names=["feature_id", "feature_name", "feature_type", "chromosome", "start", "end"]
#     )
#     # filter out sex chromosomes
#     supported_chr = [f"chr{i}" for i in range(1,23)]
#     features = features.loc[features.chromosome.isin(supported_chr), :]
#     # split features into RNA and ATAC
#     features_rna = features.copy().loc[features.feature_type == "Gene Expression", :]
#     features_rna.drop(columns=["feature_type"], inplace=True)

#     features_atac = features.copy().loc[features.feature_type == "Peaks", :]
#     features_atac.drop(columns=["feature_type"], inplace=True)

#     elapsed_time = datetime.now() - start_time
#     print(f"done! (elapsed time: {str(elapsed_time).split('.')[0]})", end="\n", flush=True)
#     return features_rna, features_atac


# def load_10x_matrix_sparse(path_x_y, index_rna, index_atac):
#     """
#     load combined scRNA and scATAC mtx file to sparse matrix, binarize the data and split it into separate scRNA and scATAC data
#     format: (barcodes * features)
#     """
#     start_time = datetime.now()
#     print("Loading RNA and ATAC data...", end=" ", flush=True)
#     file_path = os.path.join(path_x_y, "matrix.mtx")  ## scipy.io.mmread can not read .gz compressed files
#     data_rna_atac = mmread(file_path).tocsr()
#     data_rna_atac = data_rna_atac.transpose()
#     elapsed_time = datetime.now() - start_time
#     print(f"done! (elapsed time: {str(elapsed_time).split('.')[0]})", end="\n", flush=True)

#     start_time = datetime.now()
#     print("Splitting RNA and ATAC data...", end="\t", flush=True)
#     data_rna = data_rna_atac[:, index_rna]
#     data_rna = data_rna.astype(int)

#     data_atac = data_rna_atac[:, index_atac]
#     data_atac[data_atac != 0] = 1
#     data_atac = data_atac.astype(int)

#     assert data_rna.shape[0] == data_atac.shape[0], "Number of cells does not match between scRNA and scATAC!"
#     elapsed_time = datetime.now() - start_time
#     print(f"done! (elapsed time: {str(elapsed_time).split('.')[0]})", end="\n", flush=True)
#     return data_rna, data_atac


def pred_rna_norm(
    autoencoder,
    input_dim_rna,
    data_atac_test,
    batch_atac_test,
    output_prefix="",
    batch_size=16,
):
    """
    predict scRNA normalized expression from scATAC input
    Parameters
    ----------
    input_dim_rna: input scRNA gene dimension, int
    data_atac_test: scATAC expression, ncell x input_dim_atac
    batch_atac_test: scATAC batch factor, ncell x batch_dim_y
    output_prefix: output prefix

    Output
    ----------
    test_translation_rna_norm: ncell x ngenes
    """
    test_translation_rna_norm = {}
    for batch_id in range(0, data_atac_test.shape[0] // batch_size + 1):
        index_range = list(
            range(
                batch_size * batch_id,
                min(batch_size * (batch_id + 1), data_atac_test.shape[0]),
            )
        )
        test_translation_rna_norm[batch_id] = autoencoder.predict_rnanorm_translation(
            input_dim_rna,
            data_atac_test[
                index_range,
            ].todense(),
            batch_atac_test[
                index_range,
            ].todense(),
        )
        # test_translation_rna_norm[batch_id] = autoencoder.predict_rnanorm_translation(data_rna_test[index_range,].todense(), data_atac_test[index_range,].todense(), batch_rna_test[index_range,].todense(), batch_atac_test[index_range,].todense());

    test_translation_rna_norm = np.concatenate(
        [v for k, v in sorted(test_translation_rna_norm.items())], axis=0
    )
    if output_prefix == "":
        return test_translation_rna_norm
    else:
        np.savetxt(
            output_prefix + "_rnanorm_pred.txt",
            test_translation_rna_norm,
            delimiter="\t",
            fmt="%1.10f",
        )


def pred_atac_norm(
    autoencoder,
    data_rna_test,
    batch_rna_test,
    input_dim_atac,
    output_prefix="",
    atac_index="",
    batch_size=16,
):
    """
    predict scATAC normalized expression from scRNA input
    Parameters
    ----------
    data_rna_test: scRNA expression, ncell x input_dim_rna
    batch_rna_test: scRNA batch factor, ncell x batch_dim_x
    input_dim_atac: scATAC batch factor dimension, int
    output_prefix: output prefix
    atac_index: index of scATAC peak to output ('' for all peaks)

    Output
    ----------
    test_translator_reconstr_atac_norm_output: ncell x npeaks
    """
    test_translator_reconstr_atac_norm = {}
    for batch_id in range(0, data_rna_test.shape[0] // batch_size + 1):
        index_range = list(
            range(
                batch_size * batch_id,
                min(batch_size * (batch_id + 1), data_rna_test.shape[0]),
            )
        )
        test_translator_reconstr_atac_norm[
            batch_id
        ] = autoencoder.predict_atacnorm_translation(
            data_rna_test[
                index_range,
            ].todense(),
            batch_rna_test[
                index_range,
            ].todense(),
            input_dim_atac,
        )

    test_translator_reconstr_atac_norm_output = np.concatenate(
        [v for k, v in sorted(test_translator_reconstr_atac_norm.items())], axis=0
    )
    if atac_index != "":
        test_translator_reconstr_atac_norm_output = (
            test_translator_reconstr_atac_norm_output[:, atac_index]
        )
    if output_prefix == "":
        return test_translator_reconstr_atac_norm_output
    else:
        np.savetxt(
            output_prefix + "_atacnorm_pred.txt",
            test_translator_reconstr_atac_norm_output,
            delimiter="\t",
            fmt="%1.5f",
        )


def pred_embedding(
    autoencoder,
    data_rna_test,
    batch_rna_test,
    data_atac_test,
    batch_atac_test,
    output_prefix="",
    output_domain="atac",
    batch_size=16,
):
    """
    predict scATAC normalized expression from scRNA input
    Parameters
    ----------
    data_rna_test: scRNA expression, ncell x input_dim_rna
    batch_rna_test: scRNA batch factor, ncell x batch_dim_x
    data_atac_test: scATAC expression, ncell x input_dim_atac
    batch_atac_test: scATAC batch factor, ncell x batch_dim_y
    output_prefix: output prefix
    output_domain: "rna" or "atac", output the projection to the embedding layer of the selected domain, depending on which FOSCTTM is lower based on the validation set

    Output
    ----------
    test_encoded_rna: scRNA projection on scRNA AE bottleneck layer
    test_encoded_atac: scATAC projection on scATAC AE bottleneck layer
    test_translator_encoded_rna: scATAC projection on scRNA AE bottleneck layer
    test_translator_encoded_atac: scRNA projection on scATAC AE bottleneck layer
    av_fraction_rna: overall FOSCTTM score based on projection on scRNA AE bottleneck layer
    av_fraction_atac: overall FOSCTTM score based on projection on scATAC AE bottleneck layer
    sorted_fraction_1to2: FOSCTTM score for each domain 1 cell query when mapping to domain 2
    sorted_fraction_2to1: FOSCTTM score for each domain 2 cell query when mapping to domain 1

    """
    test_encoded_rna = {}
    test_translator_encoded_rna = {}
    test_encoded_atac = {}
    test_translator_encoded_atac = {}
    for batch_id in range(0, data_rna_test.shape[0] // batch_size + 1):
        index_range = list(
            range(
                batch_size * batch_id,
                min(batch_size * (batch_id + 1), data_rna_test.shape[0]),
            )
        )
        (
            test_encoded_rna[batch_id],
            test_translator_encoded_rna[batch_id],
            test_encoded_atac[batch_id],
            test_translator_encoded_atac[batch_id],
        ) = autoencoder.predict_embedding(
            data_rna_test[
                index_range,
            ].todense(),
            data_atac_test[
                index_range,
            ].todense(),
            batch_rna_test[
                index_range,
            ].todense(),
            batch_atac_test[
                index_range,
            ].todense(),
        )

    test_encoded_rna = np.concatenate(
        [v for k, v in sorted(test_encoded_rna.items())], axis=0
    )
    test_translator_encoded_rna = np.concatenate(
        [v for k, v in sorted(test_translator_encoded_rna.items())], axis=0
    )
    test_encoded_atac = np.concatenate(
        [v for k, v in sorted(test_encoded_atac.items())], axis=0
    )
    test_translator_encoded_atac = np.concatenate(
        [v for k, v in sorted(test_translator_encoded_atac.items())], axis=0
    )

    if output_prefix == "":
        av_fraction_rna = FOSCTTM(test_encoded_rna, test_translator_encoded_rna)
        av_fraction_atac = FOSCTTM(test_encoded_atac, test_translator_encoded_atac)
        return av_fraction_rna, av_fraction_atac
    else:
        ## save projectiosn on embedding space and output FOSCTTM score per cell
        if output_domain == "rna":
            sorted_fraction_1to2, sorted_fraction_2to1 = FOSCTTM(
                test_encoded_rna, test_translator_encoded_rna, output_full=True
            )
            np.savetxt(
                output_prefix + "_rna_embedding_on_rnaVAE.txt",
                test_encoded_rna,
                delimiter="\t",
                fmt="%1.5f",
            )
            np.savetxt(
                output_prefix + "_atac_translatedembedding_on_rnaVAE.txt",
                test_translator_encoded_rna,
                delimiter="\t",
                fmt="%1.5f",
            )
            np.savetxt(
                output_prefix + "_sorted_fraction_1to2_rnaVAE.txt",
                sorted_fraction_1to2,
                delimiter="\n",
                fmt="%1.5f",
            )
            np.savetxt(
                output_prefix + "_sorted_fraction_2to1_rnaVAE.txt",
                sorted_fraction_2to1,
                delimiter="\n",
                fmt="%1.5f",
            )
        if output_domain == "atac":
            sorted_fraction_1to2, sorted_fraction_2to1 = FOSCTTM(
                test_encoded_atac, test_translator_encoded_atac, output_full=True
            )
            np.savetxt(
                output_prefix + "_atac_embedding_on_atacVAE.txt",
                test_encoded_atac,
                delimiter="\t",
                fmt="%1.5f",
            )
            np.savetxt(
                output_prefix + "_rna_translatedembedding_on_atacVAE.txt",
                test_translator_encoded_atac,
                delimiter="\t",
                fmt="%1.5f",
            )
            np.savetxt(
                output_prefix + "_sorted_fraction_1to2_atacVAE.txt",
                sorted_fraction_1to2,
                delimiter="\n",
                fmt="%1.5f",
            )
            np.savetxt(
                output_prefix + "_sorted_fraction_2to1_atacVAE.txt",
                sorted_fraction_2to1,
                delimiter="\n",
                fmt="%1.5f",
            )


def eval_alignment(
    autoencoder,
    data_rna_val,
    data_atac_val,
    batch_rna_val,
    batch_atac_val,
    data_rna_test,
    data_atac_test,
    batch_rna_test,
    batch_atac_test,
    output_prefix,
    batch_size=16,
):
    """
    evaluate cross-modality cell alignment for both validation and test set, and output FOSCTTM score based on scRNA and scATAC embedding space
    Parameters
    ----------
    data_atac_test: scATAC expression, ncell x input_dim_atac
    batch_atac_test: scATAC batch factor, ncell x batch_dim_y
    output_prefix: output prefix
    """
    sim_metric_foscttm = []
    av_fraction_rna, av_fraction_atac = pred_embedding(
        autoencoder, data_rna_val, batch_rna_val, data_atac_val, batch_atac_val
    )
    sim_metric_foscttm.extend([av_fraction_rna, av_fraction_atac])
    av_fraction_rna, av_fraction_atac = pred_embedding(
        autoencoder, data_rna_test, batch_rna_test, data_atac_test, batch_atac_test
    )
    sim_metric_foscttm.extend([av_fraction_rna, av_fraction_atac])
    np.savetxt(
        output_prefix + "_stats_foscttm.txt",
        sim_metric_foscttm,
        delimiter="\n",
        fmt="%1.5f",
    )


def eval_rna_correlation(
    autoencoder,
    data_rna_val,
    data_atac_val,
    batch_atac_val,
    data_rna_test,
    data_atac_test,
    batch_atac_test,
    batch_dim_x,
    output_prefix,
    batch_size=16,
):
    """
    evaluate predicted scRNA normalized expression for both validation and test set, and return gene-wise correlation
    Parameters
    ----------
    data_atac_test: scATAC expression, ncell x input_dim_atac
    batch_atac_test: scATAC batch factor, ncell x batch_dim_y
    output_prefix: output prefix

    """
    sim_metric_rna = []
    val_translation_rna_norm = pred_rna_norm(
        autoencoder, data_rna_val.shape[1], data_atac_val, batch_atac_val
    )
    cor_gene, cor_gene_flatten, npos_gene = plot_cor_pergene(
        data_rna_val.todense(), val_translation_rna_norm, logscale=True, normlib="norm"
    )
    sim_metric_rna.extend([np.nanmean(cor_gene), cor_gene_flatten])
    test_translation_rna_norm = pred_rna_norm(
        autoencoder, data_rna_val.shape[1], data_atac_test, batch_atac_test
    )
    cor_gene, cor_gene_flatten, npos_gene = plot_cor_pergene(
        data_rna_test.todense(),
        test_translation_rna_norm,
        logscale=True,
        normlib="norm",
    )
    sim_metric_rna.extend([np.nanmean(cor_gene), cor_gene_flatten])
    np.savetxt(
        output_prefix + "_test_rna_cor.txt", cor_gene, delimiter="\n", fmt="%1.5f"
    )
    np.savetxt(
        output_prefix + "_stats_rna_cor.txt",
        sim_metric_rna,
        delimiter="\n",
        fmt="%1.5f",
    )


def eval_atac_AUROC_AUPR(
    autoencoder,
    data_rna_train,
    data_atac_train,
    data_rna_val,
    data_atac_val,
    batch_rna_val,
    data_rna_test,
    data_atac_test,
    batch_rna_test,
    batch_dim_y,
    output_prefix,
    atac_index="",
    batch_size=16,
):
    """
    evaluate predictied scATAC expression from scRNA input, on the raw binarized profile, and return peak-wise AUROC and AUPRnorm
    Parameters
    ----------
    data_rna_train: scRNA training data, ncell x input_dim_rna
    data_atac_train: scATAC training data, ncell x input_dim_atac
    data_rna_val: scRNA validation data, ncell x input_dim_rna
    data_atac_val: scATAC validation data, ncell x input_dim_atac
    data_rna_test: scRNA data, ncell x input_dim_rna
    data_atac_test: scATAC data, ncell x input_dim_atac
    batch_rna_val: scRNA batch matrix, ncell x nbatch
    batch_dim_y: nbatch in scATAC
    output_prefix: output prefix

    """
    ## first predict sequencing depth based on training set
    sim_metric_atac = []
    lib_atac_train = np.asarray(data_atac_train.todense()).sum(axis=1, keepdims=True)
    transform_libsize = RidgeCV(
        alphas=[1e-2, 1e-1, 1, 10, 100, 1000, 10000, 100000]
    ).fit(data_rna_train, lib_atac_train)
    # print(transform_libsize.alpha_)

    if atac_index != "":
        data_atac_val = data_atac_val[:, atac_index]
        data_atac_test = data_atac_test[:, atac_index]

    ## report stats in validation set
    lib_atac_val_pred = transform_libsize.predict(data_rna_val)
    val_translator_reconstr_atac_norm = pred_atac_norm(
        autoencoder,
        data_rna_val,
        batch_rna_val,
        data_atac_train.shape[1],
        "",
        atac_index,
    )
    val_translator_reconstr_atac_norm_output = (
        val_translator_reconstr_atac_norm * lib_atac_val_pred
    )
    (
        auc_peak,
        auprc_peak,
        auc_peak_flatten,
        auprc_peak_flatten,
        npos_peak,
    ) = plot_auroc_perpeak(
        data_atac_val.todense(), val_translator_reconstr_atac_norm_output
    )
    sim_metric_atac.extend(
        [
            np.nanmean(auc_peak),
            auc_peak_flatten,
            np.nanmean(auprc_peak),
            auprc_peak_flatten,
        ]
    )

    ## report stats in test set
    lib_atac_test_pred = transform_libsize.predict(data_rna_test)
    test_translator_reconstr_atac_norm = pred_atac_norm(
        autoencoder,
        data_rna_test,
        batch_rna_test,
        data_atac_train.shape[1],
        "",
        atac_index,
    )
    test_translator_reconstr_atac_norm_output = (
        test_translator_reconstr_atac_norm * lib_atac_test_pred
    )
    (
        auc_peak,
        auprc_peak,
        auc_peak_flatten,
        auprc_peak_flatten,
        npos_peak,
    ) = plot_auroc_perpeak(
        data_atac_test.todense(), test_translator_reconstr_atac_norm_output
    )
    sim_metric_atac.extend(
        [
            np.nanmean(auc_peak),
            auc_peak_flatten,
            np.nanmean(auprc_peak),
            auprc_peak_flatten,
        ]
    )
    np.savetxt(
        output_prefix + "_stats_atac_auroc_auprnorm.txt",
        sim_metric_atac,
        delimiter="\n",
        fmt="%1.5f",
    )

    ## output stats for selected peaks in test set
    (
        auc_peak,
        auprc_peak,
        auc_peak_flatten,
        auprc_peak_flatten,
        npos_peak,
    ) = plot_auroc_perpeak(
        data_atac_test.todense(), test_translator_reconstr_atac_norm_output
    )

    np.savetxt(
        output_prefix + "_test_atac_auc.txt", auc_peak, delimiter="\n", fmt="%1.5f"
    )
    np.savetxt(
        output_prefix + "_test_atac_auprc.txt", auprc_peak, delimiter="\n", fmt="%1.5f"
    )
    np.savetxt(
        output_prefix + "_test_atac_npos.txt", npos_peak, delimiter="\n", fmt="%i"
    )


def train_polarbear_model(
    outdir,
    sim_url,
    train_test_split,
    path_x,
    path_y,
    path_x_single,
    path_y_single,
    dispersion,
    embed_dim_x,
    embed_dim_y,
    nlayer,
    dropout_rate,
    learning_rate_x,
    learning_rate_y,
    learning_rate_xy,
    learning_rate_yx,
    trans_ver,
    hidden_frac,
    kl_weight,
    patience,
    nepoch_warmup_x,
    nepoch_warmup_y,
    nepoch_klstart_x,
    nepoch_klstart_y,
    batch_size,
    train,
    evaluate,
    predict,
    supported_features = {"chromosome": ["chr" + str(i) for i in range(1, 23)]},  # chr1-22
):
    """
    train/load the Polarbear model
    Parameters
    ----------
    data_rna_train: scRNA training data, ncell x input_dim_rna
    data_atac_train: scATAC training data, ncell x input_dim_atac
    data_rna_val: scRNA validation data, ncell x input_dim_rna
    data_atac_val: scATAC validation data, ncell x input_dim_atac
    data_rna_test: scRNA data, ncell x input_dim_rna
    data_atac_test: scATAC data, ncell x input_dim_atac
    batch_rna_val: scRNA batch matrix, ncell x nbatch
    batch_dim_y: nbatch in scATAC
    outdir: output directory
    train_test_split: "random" or "babel"
    path_x: scRNA co-assay (SNARE-seq) file path
    path_y: scATAC co-assay (SNARE-seq) file path
    path_x_single: scRNA single-assay file path
    path_y_single: scATAC single-assay file path
    train: "train" or "predict", train the model or just load existing model

    """
    os.system("mkdir -p " + outdir)

    if os.path.isdir(path_x[0]) and os.path.isdir(path_y[0]):
        all_samples = find_samples_from_directory(path_x) & find_samples_from_directory(path_y)
        n_batches = len(path_x)

        ## Load RNA data
        start_time = datetime.now()
        print("Reading RNA data...", end=" ", flush=True)

        matrix_list = []
        batch_list = []
        feature_list = []
        barcodes_rna = []
        for batch_id, directory in enumerate(path_x):
            sample = [sample for sample in all_samples if sample in directory.split("/")]
            if len(sample) != 1:
                raise RuntimeError(f"Sample could not be identified for {directory}")
            sample = sample[0]
            matrix, barcodes, features = load_10x_data_sparse(
                directory=directory,
                sample=sample,
                modality="RNA",
                supported_features=supported_features,
            )
            cells = matrix.shape[0]
            encoded_batch_id = np.zeros((cells, n_batches), dtype=float)
            encoded_batch_id[:, batch_id] = 1.0

            matrix_list.append(matrix)
            batch_list.append(scipy.sparse.csr_matrix(encoded_batch_id))
            feature_list.append(features)
            barcodes_rna.extend(barcodes)

        data_rna, data_rna_batch, features_rna = concatenate_samples(
            matrix_list, batch_list, feature_list, barcodes_rna
        )

        elapsed_time = datetime.now() - start_time
        print(f"done! (elapsed time: {str(elapsed_time).split('.')[0]})", end="\n", flush=True)

        ## Load ATAC data
        start_time = datetime.now()
        print("Reading ATAC data...", end=" ", flush=True)

        matrix_list = []
        batch_list = []
        feature_list = []
        barcodes_atac = []
        for batch_id, directory in enumerate(path_y):
            sample = [sample for sample in all_samples if sample in directory.split("/")]
            if len(sample) != 1:
                raise RuntimeError(f"Sample could not be identified for {directory}")
            sample = sample[0]
            matrix, barcodes, features = load_10x_data_sparse(
                directory=directory,
                sample=sample,
                modality="ATAC",
                supported_features=supported_features,
            )
            cells = matrix.shape[0]
            encoded_batch_id = np.zeros((cells, n_batches), dtype=float)
            encoded_batch_id[:, batch_id] = 1.0

            matrix_list.append(matrix)
            batch_list.append(scipy.sparse.csr_matrix(encoded_batch_id))
            feature_list.append(features)
            barcodes_atac.extend(barcodes)

        data_atac, data_atac_batch, features_atac = concatenate_samples(
            matrix_list, batch_list, feature_list, barcodes_atac
        )

        elapsed_time = datetime.now() - start_time
        print(f"done! (elapsed time: {str(elapsed_time).split('.')[0]})", end="\n", flush=True)


        ## create chr_list for ATAC VAE
        # chr_list: dictionary using chr as keys and corresponding peak index as vals
        chr_list = {}
        for chri in features_atac["chromosome"].unique():
            chr_list[int(chri[3:])] = [
                i for i, x in enumerate(features_atac["chromosome"]) if x == chri
            ]

        ## create chr_list_range for single ATAC data
        chr_list_range = features_atac.index.to_list()

        ## Unify RNA and ATAC barcodes
        samples = []

        ## Drop barcodes that do not have RNA and ATAC data
        barcodes_set = set(barcodes_rna) & set(barcodes_atac)
        barcode_list = [barcode for barcode in barcodes_rna if barcode in barcodes_set]

        rna_index = [i for i, barcode in enumerate(barcodes_rna) if barcode in barcodes_set]
        data_rna = data_rna[rna_index, :]
        data_rna_batch = data_rna_batch[rna_index, :]

        atac_index = [i for i, barcode in enumerate(barcodes_atac) if barcode in barcodes_set]
        data_atac = data_atac[atac_index, :]
        data_atac_batch = data_atac_batch[atac_index, :]

    elif os.path.isfile(path_x[0]) and os.path.isfile(path_y[0]):
        path_x = path_x[0]
        path_y = path_y[0]

        ## input peak file and filter out peaks in sex chromosomes
        # - load chromose annotation
        chr_annot = pd.read_csv(
            path_y.split("snareseq")[0] + "peaks.txt", sep=":", header=None
        )
        chr_annot.columns = ["chr", "pos"]
        chr_list = {}
        for chri in chr_annot["chr"].unique():
            if chri in supported_features["chromosome"]:
                chr_list[int(chri[3:])] = [
                    i for i, x in enumerate(chr_annot["chr"]) if x == chri
                ]

        chr_list_range = []
        for chri in chr_list.keys():
            chr_list_range += chr_list[chri]

        ## save the list of peaks
        chr_annot.iloc[chr_list_range].to_csv(
            sim_url + "_peaks.txt", index=False, sep=":", header=None
        )

        ## load scRNA mtx file to sparse matrix, binarize the data; load batch info
        # - adultbrainfull50_rna_outer_snareseq.mtx -> data_rna
        # - adultbrainfull50_rna_outer_snareseq_barcodes_dataset.mtx -> data_rna_batch
        data_rna, data_rna_batch = load_rna_file_sparse(path_x)

        ## load scATAC mtx file to sparse matrix, binarize the data; load batch info
        # - adultbrainfull50_atac_outer_snareseq.mtx -> data_atac
        # - adultbrainfull50_atac_outer_snareseq_barcodes_dataset.mtx -> data_atac_batch
        data_atac, data_atac_batch = load_atac_file_sparse(path_y, chr_list_range)

        ## load scRNA/scATAC barcode
        # - adultbrainfull50_rna_outer_snareseq_barcodes.tsv -> data_rna_barcode
        data_rna_barcode = pd.read_csv(
            path_x.split(".mtx")[0] + "_barcodes.tsv", delimiter="\t"
        )
        barcode_list = data_rna_barcode["index"].to_list()

    else:
        raise RuntimeError("Encountered unexpected data format.")

    ## ======================================
    ## define train, validation and test

    if train_test_split == "pelican":
        # use the exact train/val/test split in PELICAN
        barcode_set = set(barcode_list)
        with open("./data/NB_PCPG_train_barcodes.txt") as fp:
            train_barcode = [barcode for barcode in fp.read().splitlines() if barcode in barcode_set]

        with open("./data/NB_PCPG_val_barcodes.txt") as fp:
            val_barcode = [barcode for barcode in fp.read().splitlines() if barcode in barcode_set]

        with open("./data/NB_PCPG_test_barcodes.txt") as fp:
            test_barcode = [barcode for barcode in fp.read().splitlines() if barcode in barcode_set]
        
        train_index = [barcode_list.index(x) for x in train_barcode]
        val_index = [barcode_list.index(x) for x in val_barcode]
        test_index = [barcode_list.index(x) for x in test_barcode]

    elif train_test_split == "babel":
        # use the exact train/val/test split in BABEL
        with open("./data/babel_test_barcodes.txt") as fp:
            test_barcode = fp.read().splitlines()

        with open("./data/babel_valid_barcodes.txt") as fp:
            valid_barcode = fp.read().splitlines()

        with open("./data/babel_train_barcodes.txt") as fp:
            train_barcode = fp.read().splitlines()

        train_index = [barcode_list.index(x) for x in train_barcode]
        val_index = [barcode_list.index(x) for x in valid_barcode]
        test_index = [barcode_list.index(x) for x in test_barcode]

    elif train_test_split == "random":
        ## randomly assign 1/5 as validation and 1/5 as test set
        cv_size = data_rna.shape[0] // 5
        rand_index = list(range(data_rna.shape[0]))
        random.seed(101)
        random.shuffle(rand_index)
        all_ord_index = range(data_rna.shape[0])
        test_index = rand_index[0:cv_size]
        all_ord_index = list(set(all_ord_index) - set(range(0, cv_size)))
        random.seed(101)
        val_ord_index = random.sample(all_ord_index, cv_size)
        all_ord_index = list(set(all_ord_index) - set(val_ord_index))
        val_index = [rand_index[i] for i in val_ord_index]
        train_index = [rand_index[i] for i in all_ord_index]


    data_rna_train = data_rna[
        train_index,
    ]
    batch_rna_train = data_rna_batch[
        train_index,
    ]
    data_rna_train_co = data_rna[
        train_index,
    ]
    batch_rna_train_co = data_rna_batch[
        train_index,
    ]
    data_rna_test = data_rna[
        test_index,
    ]
    batch_rna_test = data_rna_batch[
        test_index,
    ]
    data_rna_val = data_rna[
        val_index,
    ]
    batch_rna_val = data_rna_batch[
        val_index,
    ]

    data_atac_train = data_atac[
        train_index,
    ]
    batch_atac_train = data_atac_batch[
        train_index,
    ]
    data_atac_train_co = data_atac[
        train_index,
    ]
    batch_atac_train_co = data_atac_batch[
        train_index,
    ]
    data_atac_test = data_atac[
        test_index,
    ]
    batch_atac_test = data_atac_batch[
        test_index,
    ]
    data_atac_val = data_atac[
        val_index,
    ]
    batch_atac_val = data_atac_batch[
        val_index,
    ]

    print("--------------------------------")
    print("RNA data summary")  # barcodes * features
    print("Batches:", data_rna_batch.shape[1])
    print("Genes:", data_rna.shape[1])
    print("Cells (train):", data_rna_train.shape[0])
    print("Cells (train translate):", data_rna_train_co.shape[0])
    print("Cells (validate):", data_rna_val.shape[0])
    print("Cells (test):", data_rna_test.shape[0], "\n", flush=True)

    print("ATAC data summary")  # barcodes * features
    print("Batches:", data_atac_batch.shape[1])
    print("Peaks:", data_atac.shape[1])
    print("Cells (train):", data_atac_train.shape[0])
    print("Cells (train translate):", data_atac_train_co.shape[0])
    print("Cells (validate):", data_atac_val.shape[0])
    print("Cells (test):", data_atac_test.shape[0], "\n", flush=True)

    if train == "train":
        ## save the list of genes
        print("Saving genes...", end=" ", flush=True)
        features_rna["feature_name"].to_csv(
            sim_url + "_genes.txt", index=False, header=None
        )
        print(f"done!", end="\n", flush=True)
        
        ## save the list of peaks
        # - format:   chr10:100010308-100010612
        print("Saving peaks...", end=" ", flush=True)
        features_atac["feature_name"].to_csv(
            sim_url + "_peaks.txt", index=False, header=None
        )
        print(f"done!", end="\n", flush=True)

        ## save the corresponding barcodes
        train_barcode = np.array(barcode_list)[train_index]
        valid_barcode = np.array(barcode_list)[val_index]
        test_barcode = np.array(barcode_list)[test_index]
        np.savetxt(
            sim_url + "_train_barcodes.txt", train_barcode, delimiter="\n", fmt="%s"
        )
        np.savetxt(
            sim_url + "_valid_barcodes.txt", valid_barcode, delimiter="\n", fmt="%s"
        )
        np.savetxt(
            sim_url + "_test_barcodes.txt", test_barcode, delimiter="\n", fmt="%s"
        )

        ## load single assay data
        if path_x_single != "nornasingle":
            data_rna_single, data_rna_single_batch = load_rna_file_sparse(path_x_single)
            data_rna_train = scipy.sparse.vstack((data_rna_train, data_rna_single))
            batch_rna_train = scipy.sparse.vstack(
                (batch_rna_train, data_rna_single_batch)
            )
        if path_y_single != "noatacsingle":
            data_atac_single, data_atac_single_batch = load_atac_file_sparse(
                path_y_single, chr_list_range
            )
            data_atac_train = scipy.sparse.vstack((data_atac_train, data_atac_single))
            batch_atac_train = scipy.sparse.vstack(
                (batch_atac_train, data_atac_single_batch)
            )

        ## shuffle training set
        rand_index_rna = list(range(data_rna_train.shape[0]))
        random.seed(101)
        random.shuffle(rand_index_rna)
        data_rna_train = data_rna_train[
            rand_index_rna,
        ]
        batch_rna_train = batch_rna_train[
            rand_index_rna,
        ]

        rand_index_atac = list(range(data_atac_train.shape[0]))
        random.seed(101)
        random.shuffle(rand_index_atac)
        data_atac_train = data_atac_train[
            rand_index_atac,
        ]
        batch_atac_train = batch_atac_train[
            rand_index_atac,
        ]

        ## train the model
        tf.reset_default_graph()
        autoencoder = TranslateAE(
            input_dim_x=data_rna_val.shape[1],
            input_dim_y=data_atac_val.shape[1],
            batch_dim_x=batch_rna_val.shape[1],
            batch_dim_y=batch_atac_val.shape[1],
            embed_dim_x=embed_dim_x,
            embed_dim_y=embed_dim_y,
            dispersion=dispersion,
            chr_list=chr_list,
            nlayer=nlayer,
            dropout_rate=dropout_rate,
            output_model=sim_url,
            learning_rate_x=learning_rate_x,
            learning_rate_y=learning_rate_y,
            learning_rate_xy=learning_rate_xy,
            learning_rate_yx=learning_rate_yx,
            trans_ver=trans_ver,
            hidden_frac=hidden_frac,
            kl_weight=kl_weight,
        )

        ## neptune logger
        project = "nilsmechtel/cross-modality-translation"
        neptune_run = neptune.init(
            project=project,
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4YzlkMjIwYS1mYjg1LTQxODgtOWFmZS01ZTcyNTk4YmM0ZTYifQ==",
            capture_hardware_metrics=True,
            capture_stderr=True,
            capture_stdout=True,
        )
        print("Neptune logger initialized! Project: " + project, flush=True)
        params = copy(autoencoder.__dict__)
        params["batch_size"] = batch_size
        params["patience"] = patience
        params["nepoch_warmup_x"] = nepoch_warmup_x
        params["nepoch_warmup_y"] = nepoch_warmup_y
        params["nepoch_klstart_x"] = nepoch_klstart_x
        params["nepoch_klstart_y"] = nepoch_klstart_y
        neptune_run["parameters"] = params
        neptune_run["sys/name"] = "Polarbear"

        (
            iter_list1,
            iter_list2,
            iter_list3,
            iter_list4,
            val_reconstr_atac_loss_list,
            val_kl_atac_loss_list,
            val_reconstr_rna_loss_list,
            val_kl_rna_loss_list,
            val_translat_atac_loss_list,
            val_translat_rna_loss_list,
        ) = autoencoder.train(
            data_rna_train,
            batch_rna_train,
            data_atac_train,
            batch_atac_train,
            data_rna_val,
            batch_rna_val,
            data_atac_val,
            batch_atac_val,
            data_rna_train_co,
            batch_rna_train_co,
            data_atac_train_co,
            batch_atac_train_co,
            nepoch_warmup_x,
            nepoch_warmup_y,
            patience,
            nepoch_klstart_x,
            nepoch_klstart_y,
            output_model=sim_url,
            batch_size=batch_size,
            nlayer=nlayer,
            save_model=True,
            neptune_logger=neptune_run,
        )

        neptune_run.stop()

    elif train == "predict":
        tf.reset_default_graph()
        autoencoder = TranslateAE(
            input_dim_x=data_rna_val.shape[1],
            input_dim_y=data_atac_val.shape[1],
            batch_dim_x=batch_rna_val.shape[1],
            batch_dim_y=batch_atac_val.shape[1],
            embed_dim_x=embed_dim_x,
            embed_dim_y=embed_dim_y,
            dispersion=dispersion,
            chr_list=chr_list,
            nlayer=nlayer,
            dropout_rate=dropout_rate,
            output_model=sim_url,
            learning_rate_x=learning_rate_x,
            learning_rate_y=learning_rate_y,
            learning_rate_xy=learning_rate_xy,
            learning_rate_yx=learning_rate_yx,
            trans_ver=trans_ver,
            hidden_frac=hidden_frac,
            kl_weight=kl_weight,
        )
        autoencoder.load(sim_url)

    if evaluate == "evaluate":
        ## evaluate on size-normalized scRNA true profile
        output_prefix = sim_url
        eval_rna_correlation(
            autoencoder,
            data_rna_val,
            data_atac_val,
            batch_atac_val,
            data_rna_test,
            data_atac_test,
            batch_atac_test,
            batch_rna_train.shape[1],
            output_prefix,
        )

        ## evaluate on true scATAC profile
        # we report performance on a subset of peaks that are differentially expressed across cell types based on the SNARE-seq study (Chen et al. 2019)
        chr_annot_selected = pd.read_csv(
            args.path_y.split("snareseq")[0] + "peaks_noXY_diffexp.txt", sep="\t"
        )
        atac_index = chr_annot_selected["index"].tolist()
        atac_index[:] = [int(number) - 1 for number in atac_index]
        eval_atac_AUROC_AUPR(
            autoencoder,
            data_rna[
                train_index,
            ],
            data_atac[
                train_index,
            ],
            data_rna_val,
            data_atac_val,
            batch_rna_val,
            data_rna_test,
            data_atac_test,
            batch_rna_test,
            batch_atac_train.shape[1],
            output_prefix,
            atac_index,
        )

        ## evaluate alignment
        eval_alignment(
            autoencoder,
            data_rna_val,
            data_atac_val,
            batch_rna_val,
            batch_atac_val,
            data_rna_test,
            data_atac_test,
            batch_rna_test,
            batch_atac_test,
            output_prefix,
        )

    if predict == "predict":
        output_prefix = sim_url + "_test"
        ## output normalized scRNA prediction
        pred_rna_norm(
            autoencoder,
            data_atac_test=data_atac_test,
            batch_atac_test=batch_atac_test,
            input_dim_rna=data_rna_train.shape[1],
            output_prefix=output_prefix,
        )

        ## output normalized scATAC prediction
        pred_atac_norm(
            autoencoder,
            data_rna_test=data_rna_test,
            batch_rna_test=batch_rna_test,
            input_dim_atac=data_atac_train.shape[1],
            output_prefix=output_prefix,
        )

        ## output alignment on the test set
        pred_embedding(
            autoencoder,
            data_rna_test = data_rna_test,
            batch_rna_test = batch_rna_test,
            data_atac_test = data_atac_test,
            batch_atac_test = batch_atac_test,
            output_prefix = output_prefix,
            output_domain = "rna",
        )
        pred_embedding(
            autoencoder,
            data_rna_test = data_rna_test,
            batch_rna_test = batch_rna_test,
            data_atac_test = data_atac_test,
            batch_atac_test = batch_atac_test,
            output_prefix = output_prefix,
            output_domain = "atac",
        )

        if train_test_split in ["babel", "pelican"]:
            ## output prediction on the training set, to compare with the unseen cell type prediction
            output_prefix = sim_url + "_train"
            pred_rna_norm(
                autoencoder,
                data_atac_test=data_atac_train,
                batch_atac_test=batch_atac_train,
                input_dim_rna=data_rna_train.shape[1],
                output_prefix=output_prefix,
            )
            pred_atac_norm(
                autoencoder,
                data_rna_test=data_rna_train,
                batch_rna_test=batch_rna_train,
                input_dim_atac=data_atac_train.shape[1],
                output_prefix=output_prefix,
            )

            pred_embedding(
                autoencoder,
                data_rna_test = data_rna_train,
                batch_rna_test = batch_rna_train,
                data_atac_test = data_atac_train,
                batch_atac_test = batch_atac_train,
                output_prefix = output_prefix,
                output_domain = "rna",
            )
            pred_embedding(
                autoencoder,
                data_rna_test = data_rna_train,
                batch_rna_test = batch_rna_train,
                data_atac_test = data_atac_train,
                batch_atac_test = batch_atac_train,
                output_prefix = output_prefix,
                output_domain = "atac",
            )

            ## output prediction on the training set, to compare with the unseen cell type prediction
            output_prefix = sim_url + "_val"
            pred_rna_norm(
                autoencoder,
                data_atac_test=data_atac_val,
                batch_atac_test=batch_atac_val,
                input_dim_rna=data_rna_train.shape[1],
                output_prefix=output_prefix,
            )
            pred_atac_norm(
                autoencoder,
                data_rna_test=data_rna_val,
                batch_rna_test=batch_rna_val,
                input_dim_atac=data_atac_train.shape[1],
                output_prefix=output_prefix,
            )

            pred_embedding(
                autoencoder,
                data_rna_test = data_rna_val,
                batch_rna_test = batch_rna_val,
                data_atac_test = data_atac_val,
                batch_atac_test = batch_atac_val,
                output_prefix = output_prefix,
                output_domain = "rna",
            )
            pred_embedding(
                autoencoder,
                data_rna_test = data_rna_val,
                batch_rna_test = batch_rna_val,
                data_atac_test = data_atac_val,
                batch_atac_test = batch_atac_val,
                output_prefix = output_prefix,
                output_domain = "atac",
            )

        ## output normalized scATAC prediction on the test set
        if False:
            chr_annot_selected = pd.read_csv(
                args.path_y.split("snareseq")[0] + "peaks_noXY_diffexp.txt", sep="\t"
            )
            atac_index = chr_annot_selected["index"].tolist()
            atac_index[:] = [int(number) - 1 for number in atac_index]
            pred_atac_norm(
                autoencoder,
                data_rna_test,
                batch_rna_test,
                data_atac_train.shape[1],
                output_prefix,
                atac_index,
            )

        


def main(args):
    learning_rate_x = args.learning_rate_x
    learning_rate_y = args.learning_rate_y
    learning_rate_xy = args.learning_rate_xy
    learning_rate_yx = args.learning_rate_yx
    embed_dim_x = args.embed_dim_x
    embed_dim_y = args.embed_dim_y
    dropout_rate = float(args.dropout_rate)
    nlayer = args.nlayer
    batch_size = args.batch_size
    trans_ver = args.trans_ver
    patience = args.patience
    nepoch_warmup_x = args.nepoch_warmup_x
    nepoch_warmup_y = args.nepoch_warmup_y
    nepoch_klstart_x = args.nepoch_klstart_x
    nepoch_klstart_y = args.nepoch_klstart_y
    dispersion = args.dispersion
    hidden_frac = args.hidden_frac
    kl_weight = args.kl_weight
    train_test_split = args.train_test_split
    gene_set = args.gene_set
    peak_set = args.peak_set

    sim_url = (
        args.outdir
        + "polarbear_"
        + train_test_split
        + "_"
        + dispersion
        + "_"
        + str(nlayer)
        + "l_lr"
        + str(learning_rate_y)
        + "_"
        + str(learning_rate_x)
        + "_"
        + str(learning_rate_xy)
        + "_"
        + str(learning_rate_yx)
        + "_dropout"
        + str(dropout_rate)
        + "_ndim"
        + str(embed_dim_x)
        + "_"
        + str(embed_dim_y)
        + "_batch"
        + str(batch_size)
        + "_"
        + trans_ver
        + "_improvement"
        + str(patience)
        + "_nwarmup_"
        + str(nepoch_warmup_x)
        + "_"
        + str(nepoch_warmup_y)
        + "_klstart"
        + str(nepoch_klstart_x)
        + "_"
        + str(nepoch_klstart_y)
        + "_klweight"
        + str(kl_weight)
        + "_hiddenfrac"
        + str(hidden_frac)
    )

    supported_features = {"chromosome": ["chr" + str(i) for i in range(1, 23)]}
    if gene_set is not None and peak_set is not None:
        sim_url += f"_{os.path.splitext(os.path.basename(gene_set))[0]}"
        sim_url += f"_{os.path.splitext(os.path.basename(peak_set))[0]}"

        with open(gene_set) as f:
            gene_set = f.read().splitlines()

        with open(peak_set) as f:
            peak_set = f.read().splitlines()

        supported_features["feature_name"] = gene_set + peak_set

    print(sim_url)
    train_polarbear_model(
        args.outdir,
        sim_url,
        train_test_split,
        args.path_x,
        args.path_y,
        args.path_x_single,
        args.path_y_single,
        dispersion,
        embed_dim_x,
        embed_dim_y,
        nlayer,
        dropout_rate,
        learning_rate_x,
        learning_rate_y,
        learning_rate_xy,
        learning_rate_yx,
        trans_ver,
        hidden_frac,
        kl_weight,
        patience,
        nepoch_warmup_x,
        nepoch_warmup_y,
        nepoch_klstart_x,
        nepoch_klstart_y,
        batch_size,
        args.train,
        args.evaluate,
        args.predict,
        supported_features
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optional app description")
    parser.add_argument(
        "--train_test_split",
        type=str,
        help='train/val/test split version, "random" or "babel"',
        default="random",
    )
    parser.add_argument(
        "--train",
        type=str,
        help='"train": train the model from the beginning; "predict": load existing model for downstream prediction',
        default="predict",
    )
    parser.add_argument(
        "--evaluate",
        type=str,
        help='"evaluate": evaluate translation and alignment performance on the validation set',
        default="",
    )
    parser.add_argument(
        "--predict",
        type=str,
        help='"predict": predict translation and alignment on test set',
        default="",
    )

    parser.add_argument(
        "--path_x_single",
        type=str,
        help="path of scRNA single assay data file",
        default="nornasingle",
    )
    parser.add_argument(
        "--path_y_single",
        type=str,
        help="path of scATAC single assay data file",
        default="noatacsingle",
    )
    parser.add_argument(
        "--path_x",
        nargs="+",
        type=str,
        help="path of scRNA snare-seq co-assay data file",
        default=None,
    )
    parser.add_argument(
        "--path_y",
        nargs="+",
        type=str,
        help="path of scATAC snare-seq co-assay data file",
        default=None,
    )
    # parser.add_argument(
    #     "--path_x_y",
    #     type=str,
    #     help="directory of 10x scRNA-seq and scATAC-seq co-assay data in the format of CellRanger output (barcodes.tsv ; features.tsv ; matrix.mtx)",
    #     default=None,
    # ),
    parser.add_argument(
        "--nlayer",
        type=int,
        help="number of hidden layers in encoder and decoders in neural network",
        default=2,
    )
    parser.add_argument("--outdir", type=str, help="outdir", default="./")
    parser.add_argument("--batch_size", type=int, help="batch size", default=16)
    parser.add_argument(
        "--learning_rate_x", type=float, help="scRNA VAE learning rate", default=0.001
    )
    parser.add_argument(
        "--learning_rate_y", type=float, help="scATAC VAE learning rate", default=0.0001
    )
    parser.add_argument(
        "--learning_rate_xy",
        type=float,
        help="scRNA embedding to scATAC embedding translation learning rate",
        default=0.001,
    )
    parser.add_argument(
        "--learning_rate_yx",
        type=float,
        help="scATAC embedding to scRNA embedding translation learning rate",
        default=0.001,
    )
    parser.add_argument(
        "--dropout_rate", type=float, help="dropout rate in VAE", default=0.1
    )
    parser.add_argument("--embed_dim_x", type=int, help="embed_dim_x", default=25)
    parser.add_argument("--embed_dim_y", type=int, help="embed_dim_y", default=25)
    parser.add_argument(
        "--trans_ver",
        type=str,
        help="translation layer in between embeddings, linear or 1l or 2l",
        default="linear",
    )
    parser.add_argument(
        "--patience", type=int, help="patience for early stopping", default=45
    )
    parser.add_argument(
        "--nepoch_warmup_x",
        type=int,
        help="number of epochs to take to warm up RNA VAE kl term to maximum",
        default=400,
    )
    parser.add_argument(
        "--nepoch_warmup_y",
        type=int,
        help="number of epochs to take to warm up ATAC VAE kl term to maximum",
        default=80,
    )
    parser.add_argument(
        "--nepoch_klstart_x",
        type=int,
        help="number of epochs to wait to start to warm up RNA VAE kl term",
        default=0,
    )
    parser.add_argument(
        "--nepoch_klstart_y",
        type=int,
        help="number of epochs to wait to start to warm up ATAC VAE kl term",
        default=0,
    )
    parser.add_argument(
        "--dispersion",
        type=str,
        help="estimate dispersion per gene&batch: genebatch or per gene&cell: genecell",
        default="genebatch",
    )
    parser.add_argument(
        "--hidden_frac",
        type=int,
        help="shrink intermediate dimension by dividing this term",
        default=2,
    )
    parser.add_argument(
        "--kl_weight", type=float, help="weight of kl loss in beta-VAE", default=1
    )
    parser.add_argument(
        "--gene_set", type=str, help="path to file with subset of genes", default=None
    )
    parser.add_argument(
        "--peak_set", type=str, help="path to file with subset of peaks", default=None
    )


    args = parser.parse_args()
    main(args)

# %%

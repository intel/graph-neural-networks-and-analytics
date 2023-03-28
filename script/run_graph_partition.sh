#!/bin/bash

gnn_tmp=$1
num_parts=$2
partPath=$3

python ${WORKSPACE}/src/partition_tabformer_homo.py --CSVDataset ${gnn_tmp}/sym_tabformer_hetero_CSVDataset --num_part ${num_parts} --partition_out ${partPath}

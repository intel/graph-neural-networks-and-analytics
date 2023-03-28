#!/bin/bash

#${PROC_DATA} ${GNN_TMP} ${PART_DIR} ${NUMPART} ${OUT_DIR} ${MAP_SAVE}

PROCESSED_DATA=$1
GNN_TMP=$2
PART_DIR=$3
NUMPART=$4
OUT_DIR=$5
MAP_SAVE=$6
CONDA_ENV=$7

eval "$(conda shell.bash hook)"
conda activate dgl1.0
python --version

PY_EXEC=${CONDA_ENV}/bin/python3
echo ${PY_EXEC}
EXEC_SCRIPT=${WORKSPACE}/src/train_dist_unsupervised_transductive_fraud.py

#tabformer CSVDataset
CSVDATASET=${GNN_TMP}/sym_tabformer_hetero_CSVDataset

#model, embb and final csv outputs
mkdir -p ${OUT_DIR}
MODEL_OUT=${OUT_DIR}/tabformer_graphsage_2L_64.pt
NEMB_OUT=${OUT_DIR}/node_emb.pt #these are the default names
NEMB_OUT_MAPPED=${OUT_DIR}/node_emb_mapped.pt #these are the default names
OUT_DATA=${OUT_DIR}/tabformer_with_gnn_emb.csv

GRAPH_NAME=( "tabformer_full_homo" )

#partition data directory
NUMPART=2
CURR_PART_DIR=$PART_DIR/tabformer_${NUMPART}parts
PART_CONFIG=$CURR_PART_DIR/${GRAPH_NAME}.json

#ip_config path
IP_CONFIG=${WORKSPACE}/ip_config.txt

# Folder and filename where you want your logs.
logdir=${WORKSPACE}/logs
mkdir -p $logdir
logname=log_${GRAPH_NAME}_${NUMPART}n_$RANDOM
echo $logname
set -x


# minibatch size on each host
MB_SIZE=1000

# minibatch size on each host
HIDDEN_FS=64

# fanout per layer
FANOUT="10,15"

# num epochs to run for
EPOCHS=10

#DGL distributed configuration
NTRAINER=1
NSAMPLER=0
NSERVER=1


# launch distributed training and sstdout stored in /logdir/logname.out
python -u ${WORKSPACE}/src/launch.py --workspace ${WORKSPACE} --num_trainers ${NTRAINER} --num_samplers ${NSAMPLER} --num_servers ${NSERVER} --part_config ${PART_CONFIG} --ip_config ${IP_CONFIG} "${PY_EXEC} ${EXEC_SCRIPT} --graph_name ${GRAPH_NAME} --ip_config ${IP_CONFIG} --part_config ${PART_CONFIG} --num_epoch ${EPOCHS}  --num_hidden ${HIDDEN_FS} --fan_out ${FANOUT} --batch_size ${MB_SIZE} --CSVDataset_dir ${CSVDATASET} --remove_edge --model_out ${MODEL_OUT} --nemb_out ${NEMB_OUT}" |& tee ${logdir}/${logname}.txt

if [[ ${MAP_SAVE} == "true" ]]; then
  echo "Mapping to original graph ids followed by mapping to CSV file output"
  echo "This may take a while"
  python ${WORKSPACE}/src/map_emb.py --processed_data_path ${PROCESSED_DATA} --partition_data_path ${CURR_PART_DIR} --model_emb_path ${OUT_DIR} --out_data ${OUT_DATA}
fi

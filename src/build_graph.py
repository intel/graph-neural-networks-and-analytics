import pandas as pd
import numpy as np
import csv

import time
import yaml
import os
import argparse


def main(args):

    CSVDataset_dir = os.path.join(args.gnn_tmp, "sym_tabformer_hetero_CSVDataset")
    for dir in [args.gnn_tmp, CSVDataset_dir]:
        if not os.path.exists(dir):
            os.mkdir(dir)

    # 1.   load CSV file output of Classical ML edge featurization workflow
    print("loading processed data")
    start = time.time()
    df = pd.read_csv(args.data_in)
    t_load_data = time.time()
    print("time lo load processed data", t_load_data - start)

    # 2.   Renumbering - generating node/edge ids starting from zero
    print("node renumbering, mapping and generating train/val/test masks from splits")

    def column_index(series, offset=0):
        return {k: v + offset for v, k in enumerate(series.value_counts().index.values)}

    # hetero mapping (each type starts from zero)
    card2idx = column_index(
        df["card_id"]
    )  # {v: idx for idx, v in enumerate(data['card_id'].values)}
    merchant2idx = column_index(df["merchant_id"])
    # homo mapping (starts from zero to total_num_nodes)
    offset = len(card2idx)
    merchant2idx_h = column_index(df["merchant_id"], offset=offset)

    # write homo node mapping
    df_cards = pd.DataFrame.from_dict(card2idx, orient="index")
    df_merch_h = pd.DataFrame.from_dict(merchant2idx_h, orient="index")
    df_mapping = pd.concat([df_cards, df_merch_h])
    df_mapping.to_csv(os.path.join(args.gnn_tmp, "node_mapping_homo.csv"), header=None)

    # add new ids to dataframe
    # to create a DGL graph all nodes types need to be renamed starting from 0. we need this dictionaries to map them back
    df["cardIdx"] = df["card_id"].map(card2idx)
    df["merchIdx"] = df["merchant_id"].map(merchant2idx)
    df["merchIdx_new"] = df["merchant_id"].map(merchant2idx_h)

    # 3    create masks for train, val and test splits (add columns with masks)
    # splits already sabed in edge_featurized input because we are doing target encoding and want to avoid data leakage
    oneh_enc_cols = ["split"]
    df = pd.concat(
        [df, pd.get_dummies(df[oneh_enc_cols].astype("category"), prefix="masks")],
        axis=1,
    )
    t_renum = time.time()
    print("time to renum and split train/val/tst masks", t_renum - t_load_data)

    # 4    Prepare CSVDataset files for DGL to ingest and create graph
    print("Writting data into set of CSV files (nodes/edges)")

    # write meta.yaml
    meta_yaml = """
dataset_name: sym_tabformer_hetero_CSVDataset
edge_data:
- file_name: edges_0.csv
  etype: [cardIdx, transaction, merchIdx]
- file_name: edges_1.csv
  etype: [merchIdx, sym_transaction, cardIdx]
node_data:
- file_name: nodes_0.csv
  ntype: cardIdx
- file_name: nodes_1.csv
  ntype: merchIdx
    """

    meta = yaml.safe_load(meta_yaml)

    with open(os.path.join(CSVDataset_dir, "meta.yaml"), "w") as file:
        yaml.dump(meta, file)

    # Note: feat_as_str needs to be a string of comma separated values enclosed in double quotes for dgl default parser to work
    keysList = list(df.keys())
    unwantedkeys = [
        "user",
        "card",
        "merchant_name",
        "card_id",
        "merchant_id",
        "cardIdx",
        "merchIdx",
        "merchIdx_new",
        "is_fraud?",
        "split",
        "masks_0",
        "masks_1",
        "masks_2",
    ]
    feat_keys = [k for k in keysList if k not in unwantedkeys]
    print("features for CSSVDataset edges: ", feat_keys)
    df["feat_as_str"] = df[feat_keys].astype(str).apply(",".join, axis=1)

    # write 2 edge files
    df[
        [
            "cardIdx",
            "merchIdx",
            "is_fraud?",
            "masks_0",
            "masks_1",
            "masks_2",
            "feat_as_str",
        ]
    ].to_csv(
        os.path.join(CSVDataset_dir, "edges_0.csv"),
        index=False,
        header=[
            "src_id",
            "dst_id",
            "label",
            "train_mask",
            "val_mask",
            "test_mask",
            "feat",
        ],
    )
    df[
        [
            "merchIdx",
            "cardIdx",
            "is_fraud?",
            "masks_0",
            "masks_1",
            "masks_2",
            "feat_as_str",
        ]
    ].to_csv(
        os.path.join(CSVDataset_dir, "edges_1.csv"),
        index=False,
        header=[
            "src_id",
            "dst_id",
            "label",
            "train_mask",
            "val_mask",
            "test_mask",
            "feat",
        ],
    )

    # write 2 nodes files - featuresless nodes, no label
    np.savetxt(
        os.path.join(CSVDataset_dir, "nodes_0.csv"),
        df["cardIdx"].unique(),
        delimiter=",",
        header="node_id",
        comments="",
    )
    np.savetxt(
        os.path.join(CSVDataset_dir, "nodes_1.csv"),
        df["merchIdx"].unique(),
        delimiter=",",
        header="node_id",
        comments="",
    )
    t_csvDataset = time.time()
    print("time to write CSVDatasets", t_csvDataset - t_renum)


def file(raw_path):
    if not os.path.isfile(raw_path):
        raise argparse.ArgumentTypeError(
            '"{}" is not an existing file'.format(raw_path)
        )
    return os.path.abspath(raw_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BuildGraph")
    parser.add_argument(
        "--data_in",
        type=file,
        default="/DATA_IN/processed_data.csv",
        help="Input file with path (processed_data.csv) ",
    )
    parser.add_argument(
        "--gnn_tmp", default="/GNN_TMP/", help="The path to the gnn_tmp"
    )

    args = parser.parse_args()

    print(args)
    main(args)

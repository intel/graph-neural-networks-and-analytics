import torch
import pandas as pd
import time
import argparse
import os


def main(args):
    IN_DATA = args.processed_data_path
    NODE_EMB = args.model_emb_path + "/" + args.node_emb_name + ".pt"
    NODE_EMB_MAPPED = args.model_emb_path + "/" + args.node_emb_name + "_mapped.pt"
    NMAP = os.path.join(args.partition_data_path, "nmap.pt")
    OUT_DATA = args.out_data_path

    # 1.   load CSV file output of Classical ML edge featurization workflow
    print("loading processed data")
    start = time.time()
    df = pd.read_csv(IN_DATA)
    t_load_data = time.time()
    print("time lo load processed data", t_load_data - start)

    # 2.   Renumbering - generating node/edge ids starting from zero
    print("node renumbering, mapping and generating train/val/test masks from splits")

    def column_index(series, offset=0):
        return {k: v + offset for v, k in enumerate(series.value_counts().index.values)}

    card2idx = column_index(
        df["card_id"]
    )  # {v: idx for idx, v in enumerate(data['card_id'].values)}
    offset = len(card2idx)
    merchant2idx_h = column_index(df["merchant_id"], offset=offset)
    df_cards = pd.DataFrame.from_dict(card2idx, orient="index")
    df_merch_h = pd.DataFrame.from_dict(merchant2idx_h, orient="index")

    # add new ids to dataframe
    # to create a DGL graph all nodes types need to be renamed starting from 0. we need this dictionaries to map them back
    df["cardIdx"] = df["card_id"].map(card2idx)
    df["merchIdx_new"] = df["merchant_id"].map(merchant2idx_h)

    # 3.   load node embeddings from file, add them to edge features and save file for Classic ML workflow (since model is trained as homo, no mapping needed.)
    print("Loading embeddings from file and adding to preprocessed CSV file")
    node_emb = torch.load(NODE_EMB)

    # 4 map from  partition to global
    print("mapping from partition ids to full graph ids")
    nmap = torch.load(NMAP)

    orig_node_emb = torch.zeros(node_emb.shape, dtype=node_emb.dtype)
    orig_node_emb[nmap] = node_emb
    torch.save(orig_node_emb, NODE_EMB_MAPPED)

    node_emb_arr = orig_node_emb.cpu().detach().numpy()
    # node_emb_arr = node_emb.cpu().detach().numpy()
    node_emb_dict = {i: val for i, val in enumerate(node_emb_arr)}
    t_load_emb = time.time()
    print("Time to load emb", t_load_emb - t_load_data)

    card_emb = pd.DataFrame(df["cardIdx"].map(node_emb_dict).tolist()).add_prefix("c")
    merch_emb = pd.DataFrame(df["merchIdx_new"].map(node_emb_dict).tolist()).add_prefix(
        "m"
    )
    df = df.join([card_emb, merch_emb])
    df.drop(
        columns=[
            "cardIdx",
            "merchIdx_new",
        ],
        axis=1,
        inplace=True,
    )  # we dont need it for out CSV, features already in df
    print("CSV output shape: ", df.shape)
    print(df.shape)
    print(df.head(2))

    df.to_csv(OUT_DATA, index=False)
    print(
        "Time to append node embeddings to edge features CSV", time.time() - t_load_emb
    )


def directory(raw_path):
    if not os.path.isdir(raw_path):
        raise argparse.ArgumentTypeError(
            '"{}" is not an existing directory'.format(raw_path)
        )
    return os.path.abspath(raw_path)


def file(raw_path):
    if not os.path.isfile(raw_path):
        raise argparse.ArgumentTypeError(
            '"{}" is not an existing file'.format(raw_path)
        )
    return os.path.abspath(raw_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MapEmb and save")
    parser.add_argument(
        "--processed_data_path", type=file, help="The path to the processed_data.csv"
    )
    parser.add_argument(
        "--partition_data_path", type=directory, help="The path to the partition folder"
    )
    parser.add_argument(
        "--model_emb_path",
        type=directory,
        help="The path to the pt files generated in training",
    )
    parser.add_argument(
        "--node_emb_name",
        type=str,
        default="node_emb",
        help="The path to the node embedding file",
    )
    parser.add_argument(
        "--out_data_path",
        type=str,
        help="The path to the csv data file with mapped node embeddings",
    )
    args = parser.parse_args()

    print(args)
    main(args)

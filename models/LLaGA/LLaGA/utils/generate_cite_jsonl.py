"""
Generate JSONL task files for CITE dataset (node classification).
Usage:
    python utils/generate_cite_jsonl.py
Output:
    dataset/cite/sampled_2_10_train.jsonl
    dataset/cite/sampled_2_10_test.jsonl
    dataset/cite/sampled_2_10_val.jsonl
"""

import json
import os
import torch
from tqdm import tqdm
from utils.data_process import get_fix_shape_subgraph_sequence_fast, generate_edge_list
from utils.constants import DEFAULT_GRAPH_TOKEN

USE_HOP = 2
SAMPLE_SIZE = 10
DATA_PATH = "dataset/cite/CITE.pt"
OUT_DIR = "dataset/cite"

NC_PROMPT = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent papers and edges represent citation relationships. Please classify the center node into one of the given categories: {{label_list}}. Which category does the center node belong to?"


def write_jsonl(path, samples):
    with open(path, 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + '\n')
    print(f"Saved {len(samples)} samples to {path}")


def main():
    data = torch.load(DATA_PATH, map_location='cpu')
    print(f"Loaded data: {data}")

    # build adjacency list
    print("Building edge list...")
    edge_list = generate_edge_list(data)

    label_list = ", ".join(data.label_texts)
    prompt = NC_PROMPT.format(label_list=label_list)

    splits = {
        "train": data.train_mask,
        "val":   data.val_mask,
        "test":  data.test_mask,
    }

    for split_name, mask in splits.items():
        node_ids = mask.nonzero(as_tuple=True)[0].tolist()
        samples = []
        for node_id in tqdm(node_ids, desc=f"Processing {split_name}"):
            subgraph = get_fix_shape_subgraph_sequence_fast(
                edge_list, node_id, k_hop=USE_HOP, sample_size=SAMPLE_SIZE
            )
            label = data.label_texts[data.y[node_id].item()]
            sample = {
                "id": node_id,
                "graph": [subgraph],
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt",   "value": f"The center node belongs to {label}."},
                ]
            }
            samples.append(sample)

        out_path = os.path.join(OUT_DIR, f"sampled_{USE_HOP}_{SAMPLE_SIZE}_{split_name}.jsonl")
        write_jsonl(out_path, samples)


if __name__ == "__main__":
    main()

import os
import json
import torch
import numpy as np
import pandas as pd
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T

def get_raw_text_arxiv(use_text=False, seed=0):
    data = torch.load('/mnt/data/zch/glbench/datasets/arxiv.pt')
    text = data.raw_texts
    if data.train_mask.dim() == 10:
        data.train_mask = data.train_mask[0]
        data.val_mask = data.val_mask[0]
        data.test_mask = data.test_mask[0]
    return data, text

def save_tensor_as_csv(tensor, file_path):
    df = pd.DataFrame(tensor.numpy())  # Convert tensor to numpy array and then to DataFrame
    df.to_csv(file_path, index=False)

def save_data(data, text):
    output_dir = 'output_data'  # Root folder to save all the data
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over the attributes of 'data' (which is a torch geometric data object)
    for attribute, value in data.items():
        # Create a folder for each attribute if it doesn't exist
        attribute_dir = os.path.join(output_dir, attribute)
        if not os.path.exists(attribute_dir):
            os.makedirs(attribute_dir)

        # Save tensor data (x, y, etc.) to CSV
        if isinstance(value, torch.Tensor):
            file_path = os.path.join(attribute_dir, f'{attribute}.csv')
            save_tensor_as_csv(value, file_path)
        
        # Save raw_texts (if applicable) as JSON
        elif isinstance(value, list):
            file_path = os.path.join(attribute_dir, f'{attribute}.json')
            with open(file_path, 'w') as f:
                json.dump(value, f, indent=4)

    # Save text data as JSON
    text_dir = os.path.join(output_dir, 'raw_texts')
    if not os.path.exists(text_dir):
        os.makedirs(text_dir)
    with open(os.path.join(text_dir, 'raw_texts.json'), 'w') as f:
        json.dump(text, f, indent=4)

def main():
    data, text = get_raw_text_arxiv(use_text=False, seed=0)
    
    # Save data to files
    save_data(data, text)

main()

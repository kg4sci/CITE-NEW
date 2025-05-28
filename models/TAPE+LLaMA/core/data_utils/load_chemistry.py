import os
import json
import torch
import numpy as np
import pandas as pd
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T

def get_raw_text_chemistry(use_text=False, seed=0):
    # Load the chemistry dataset
    data = torch.load('/mnt/data/zch/glbench/datasets/chemistry.pt')  # Update the path accordingly
    text = data.raw_texts  # Assuming the dataset has 'raw_texts' similar to arxiv

    # Ensure that the data masks have the correct shape (in case it's a list of tensors or dimensions)
    # Create new masks based on random splitting (80% train, 10% validation, 10% test)***********重写mask方式
    num_nodes = data.num_nodes  # Total number of nodes (samples) in the graph
    
    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    # Generate a random permutation of node indices
    indices = torch.randperm(num_nodes)

    # Calculate the split sizes (80%, 10%, 10%)
    train_size = int(0.8 * num_nodes)
    val_size = int(0.1 * num_nodes)
    test_size = num_nodes - train_size - val_size  # The remaining will be the test set

    # Create masks based on the random permutation of indices
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # Assign the random split to train, validation, and test sets
    data.train_mask[indices[:train_size]] = True
    data.val_mask[indices[train_size:train_size + val_size]] = True
    data.test_mask[indices[train_size + val_size:]] = True

    # Return the data and text
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
    data, text = get_raw_text_chemistry(use_text=False, seed=0)
    
    # Save data to files
    save_data(data, text)

main()

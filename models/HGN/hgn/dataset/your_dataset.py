import dgl
import torch as th
import torch
import pandas as pd
import numpy as np
from .utils import get_binary_mask


class YourDataset:
    def __init__(self, name, raw_dir=''):
        self.name = name
        self.raw_dir = raw_dir
        self.graph, self.category, self.num_classes, self.in_dim = self.load_data()

    def load_data(self):
        # 加载节点信息
        # 读取节点文件
        paper_nodes = pd.read_csv("/root/autodl-tmp/dgl/1openHGNN-main/openhgnn/dataset/data/chemistry/nodes/128/papertitle_embedding128.csv")  # paper_id, title, feature1, feature2, label
        author_nodes = pd.read_csv("/root/autodl-tmp/dgl/1openHGNN-main/openhgnn/dataset/data/chemistry/nodes/128/author_embedding128.csv")  # author_id, name, feature1, label
        keywords_nodes = pd.read_csv("/root/autodl-tmp/dgl/1openHGNN-main/openhgnn/dataset/data/chemistry/nodes/128/keywords_embedding128.csv")  # keywords_id, word
        journal_nodes = pd.read_csv("/root/autodl-tmp/dgl/1openHGNN-main/openhgnn/dataset/data/chemistry/nodes/128/journal_embedding128.csv")  # venue_id, name
        paper_labels = pd.read_csv("/root/autodl-tmp/dgl/1openHGNN-main/openhgnn/dataset/data/chemistry/nodes/128/id+label.csv")
        
        # 加载边信息
        paper_author_edges = pd.read_csv("/root/autodl-tmp/dgl/1openHGNN-main/openhgnn/dataset/data/chemistry/edges/author_paper_new.csv")  # paper_id, author_id
        paper_journal_edges = pd.read_csv("/root/autodl-tmp/dgl/1openHGNN-main/openhgnn/dataset/data/chemistry/edges/paper_journal.csv")  # paper_id, venue_id
        paper_keywords_edges = pd.read_csv("/root/autodl-tmp/dgl/1openHGNN-main/openhgnn/dataset/data/chemistry/edges/paper_keywords.csv")  # paper_id, keywords_id
        paper_paper_edges = pd.read_csv("/root/autodl-tmp/dgl/1openHGNN-main/openhgnn/dataset/data/chemistry/edges/paper_paper.csv")  # paper_id, paper_id
        
        # 创建异构图
        g = dgl.heterograph({
            ('paper', 'pa', 'author'): (paper_author_edges['paper_new_id'].values, paper_author_edges['author_new_id'].values),
            ('author', 'ap', 'paper'): (paper_author_edges['author_new_id'].values, paper_author_edges['paper_new_id'].values),
            ('paper', 'pk', 'keywords'): (paper_keywords_edges['paper_id'].values, paper_keywords_edges['keywords_id'].values),
            ('keywords', 'kp', 'paper'): (paper_keywords_edges['keywords_id'].values, paper_keywords_edges['paper_id'].values),
            ('paper', 'pj', 'journal'): (paper_journal_edges['paper_id'].values, paper_journal_edges['journal_id'].values),
            ('journal', 'jp', 'paper'): (paper_journal_edges['journal_id'].values, paper_journal_edges['paper_id'].values),
            ('paper', 'pp', 'paper'): (paper_paper_edges['source'].values, paper_paper_edges['target'].values)
        })
        

        # 设置节点特征
        # 假设你需要选择前128个embedding特征
        author_columns = [f'name_embedding_{i}' for i in range(128)]
        paper_columns = [f'title_embedding_{i}' for i in range(128)]
        keywords_columns = [f'term_embedding_{i}' for i in range(128)]
        journal_columns = [f'journal_embedding_{i}' for i in range(128)]


        # paper_columns = [f'title_embedding_{i}' for i in range(128)]

        # 使用iloc选择这些列
        g.nodes['author'].data['h'] = torch.tensor(author_nodes[author_columns].values, dtype=torch.float32)
        g.nodes['paper'].data['h'] = torch.tensor(paper_nodes[paper_columns].values, dtype=torch.float32)
        g.nodes['keywords'].data['h'] = torch.tensor(keywords_nodes[keywords_columns].values, dtype=torch.float32)
        g.nodes['journal'].data['h'] = torch.tensor(journal_nodes[journal_columns].values, dtype=torch.float32)
        # 如果有多个节点类型，检查其他节点类型的特征

        

        # 准备标签
        labels = th.LongTensor(paper_labels['label'].values)

        
        # 划分训练、验证和测试集
        num_nodes = g.number_of_nodes('paper') 
        train_idx, val_idx, test_idx = split_indices(num_nodes)
        train_mask = get_binary_mask(num_nodes, train_idx)
        val_mask = get_binary_mask(num_nodes, val_idx)
        test_mask = get_binary_mask(num_nodes, test_idx)
        
        g.nodes['paper'].data['labels'] = labels
        g.nodes['paper'].data['train_mask'] = train_mask
        g.nodes['paper'].data['val_mask'] = val_mask
        g.nodes['paper'].data['test_mask'] = test_mask
        
        return g, 'paper', 106, g.nodes['paper'].data['h'].shape[1]
    
def split_indices(num_nodes, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    # """随机划分训练集、验证集和测试集"""
        assert train_ratio + val_ratio + test_ratio == 1.0, "The sum of ratios must be 1"

        all_indices = np.random.permutation(num_nodes)
        
        train_size = int(num_nodes * train_ratio)
        val_size = int(num_nodes * val_ratio)
        
        train_idx = all_indices[:train_size]
        val_idx = all_indices[train_size:train_size+val_size]
        test_idx = all_indices[train_size+val_size:]
        
        return train_idx, val_idx, test_idx
#### HGN models

```
pip install -r requirements.txt
pip install dgl -f https://data.dgl.ai/wheels/repo.html
pip install openhgnn
```

#### Datasets

download [here](https://huggingface.co/datasets/kg4sci/CITE/tree/main) and put it on ./hgn/dataset/data

#### Run

python main.py -m **model** -t node_classification -d my_custom_node_classification -g 0 --use_best_config

model optional:
RGCN SimpleHGN HGT NARS CompGCN HPN
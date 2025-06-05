# CITE - A Comprehensive Benchmark for Heterogeneous Text-Attributed Graphs on Catalytic Materials

## 0. Python environment requirements

- Python  >= 3.6
- PyTorch >= 2.1.0
- CUDA >= 11.8

## 1.Download CITE

| Dataset version | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| csv             | csv files include node and edge informations, including raw-texts and 128-dimention embedding<br />Download [paper](https://huggingface.co/datasets/kg4sci/CITE/blob/main/paper_embedding.csv), [author](https://huggingface.co/datasets/kg4sci/CITE/blob/main/author_embedding128.csv), [journal](https://huggingface.co/datasets/kg4sci/CITE/blob/main/journal_embedding128.csv), [keywords](https://huggingface.co/datasets/kg4sci/CITE/blob/main/keywords_embedding128.csv), [label](https://huggingface.co/datasets/kg4sci/CITE/blob/main/paper_label.csv) and move them to `datasets/csv/node`<br /> Download [paper-paper](https://huggingface.co/datasets/kg4sci/CITE/blob/main/paper_paper.csv), [paper-author](https://huggingface.co/datasets/kg4sci/CITE/blob/main/author_paper.csv), [paper-journal](https://huggingface.co/datasets/kg4sci/CITE/blob/main/paper_journal.csv), [paper-keywords](https://huggingface.co/datasets/kg4sci/CITE/blob/main/paper_keywords.csv) and move them to `datasets/csv/edge` |
| pt              | [chemistry.pt](https://huggingface.co/datasets/kg4sci/CITE/blob/main/chemistry.pt) and [my_graph_data.pt](https://huggingface.co/datasets/kg4sci/CITE/blob/main/my_graph_data.pt) provides all information of  node and edges.<br /> Download the datasets and move them to `datasets/pt/` |
| json            | json file include manufactured for GraphGPT, which includs two training stages and one eval stage.<br />Download [stage1](https://huggingface.co/datasets/kg4sci/CITE/blob/main/stage1.json), [stage2](https://huggingface.co/datasets/kg4sci/CITE/blob/main/stage2.json) and [eval](https://huggingface.co/datasets/kg4sci/CITE/blob/main/eval_std.json) and move them to `dataset/json/` |

## 2.Run different models

### Homogeneous Graph models

#### Requirements

```
cd models/gnn/
conda create --name gnn python==3.10
conda activate gnn
pip install -r requirements.txt

```

#### How to run

```
chomd +x run.sh
./run.sh
```

### Heterogeneous Graph models

#### Requirements

```
cd models/HGN/
conda create --name hgn python==3.8
conda activate hgn
pip install -r requirements.txt
pip install dgl -f https://data.dgl.ai/wheels/repo.html
pip install openhgnn
```

#### How to run

```
python main.py -m model name -t node_classification -d my_custom_node_classification -g 0 --use_best_config
```

model names:
RGCN SimpleHGN HGT NARS CompGCN HPN

### LLM

#### Requirements

```
cd models/TAPE+LLaMA
conda creative --name LLM python==3.10
conda activate LLM
pip install -r requirements.txt
```

#### Run

python chat.py

#### Evaluation

python llama_preds.py

### LLM+Graph models

#### TAPE

##### Requirements

```
cd models/TAPE+LLaMA
conda create --name TAPE python==3.10
conda activate TAPE
pip install -r requirements.txt
```

##### How to run

```
chomd +x run.sh
./run.sh
```

#### GraphGPT

##### Requirements

```
cd models/GraphGPT
conda create --name GraphGPT python==3.10
conda activate GraphGPT
pip install -r requirements.txt
```

##### How to run

###### Stage-1

```
chmod +x stage1.sh
./stage1.sh
```

###### Extract

```
chmod +x extract.sh
./extract.sh
```

###### Stage-2

```
xchmod +x stage2.sh
./stage2.sh
```

###### **Eval**

```
chmod +x eval.sh
./eval.sh
```


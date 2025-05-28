## GraphGPT

### Environment setup

```
pip install -r requirements.txt
```

#### Datasets

download [my_graph_data.pt](https://huggingface.co/datasets/kg4sci/CITE/blob/main/my_graph_data.pt) and put in ./graph_data/my_graph_data_new.pt

download [stage1.json](https://huggingface.co/datasets/kg4sci/CITE/blob/main/stage1.json) and put in ./data/stage_1/stage1.json

download [stage2.json](https://huggingface.co/datasets/kg4sci/CITE/blob/main/stage2.json) and put in ./data/stage_2/stage2.json

download [eval_std.json](https://huggingface.co/datasets/kg4sci/CITE/blob/main/eval_std.json) and put in ./eval/eval_std.json

#### Run

**Stage-1**

```
chmod +x stage1.sh
./stage1.sh
```

**Extract**

```
chmod +x extract.sh
./extract.sh
```

**Stage-2**

```
xchmod +x stage2.sh
./stage2.sh
```

**Eval**

```
chmod +x eval.sh
./eval.sh
```


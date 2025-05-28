## TAPE+LLaMA

#### Environment setup

```
pip install -r requirements.txt
```

#### Datasets

download [here](https://huggingface.co/datasets/kg4sci/CITE/blob/main/chemistry.pt) and put it on ../../dataset

### For TAPE

#### Run

```
chmod +x run.sh
./run.sh
```

### For LLaMA

### Run

python chat.py

### Eval

python llama_preds.py

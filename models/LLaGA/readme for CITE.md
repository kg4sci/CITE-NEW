# LLaGA on Chemistry Dataset

This document describes how to use LLaGA with the Chemistry dataset.
Follow the original README for environment setup (Step 1), then use this guide instead of Steps 2-4.

---

## Step 2: Data Preparation

Place the chemistry dataset under `./dataset/chemistry/`:

```
dataset/
└── chemistry/
    ├── chemistry.pt                 # main graph data (must contain data.label_texts and data.y)
    ├── sampled_2_10_train.jsonl     # node classification training prompts
    ├── sampled_2_10_test.jsonl      # node classification test prompts
    └── sbert_x.pt                  # (optional) pre-computed SBERT node embeddings
```

**Notes on `chemistry.pt`:**
- `data.label_texts`: list of 107 category names (e.g. `ENGINEERING`, `MATERIALSSCIENCE`, ...)
- `data.y`: node label indices, shape `[num_nodes]`
- `data.x`: raw node features (used when `--pretrained_embedding_type chemistry`)
- `data.edge_index`: graph edges in COO format

---

## Step 3: Training

```shell
# single GPU
CUDA_VISIBLE_DEVICES=0 ./scripts/train.sh vicuna nc chemistry 16 chemistry

# multiple GPUs (deepspeed)
./scripts/train_deepspeed.sh vicuna nc chemistry 16 chemistry
```

**Arguments:**
- `$1` model type: `vicuna` / `vicuna_4hop` / `llama` etc.
- `$2` task: `nc` (node classification)
- `$3` dataset: `chemistry`
- `$4` batch size: `16`
- `$5` embedding type: `chemistry` (uses `data.x` directly) or `sbert` / `roberta`

---

## Step 4: Evaluation

```shell
model_path="/path/to/projector"      # local path or huggingface repo
model_base="lmsys/vicuna-7b-v1.5-16k"
mode="v1"
dataset="chemistry"
task="nc"
emb="chemistry"                      # or sbert / roberta
use_hop=2
sample_size=10
template="ND"
output_path="./results/chemistry_nc.jsonl"

python eval/eval_pretrain.py \
  --model_path ${model_path} \
  --model_base ${model_base} \
  --conv_mode ${mode} \
  --dataset ${dataset} \
  --pretrained_embedding_type ${emb} \
  --use_hop ${use_hop} \
  --sample_neighbor_size ${sample_size} \
  --answers_file ${output_path} \
  --task ${task} \
  --cache_dir ../../checkpoint \
  --template ${template}
```

Then compute metrics:

```shell
python eval/eval_res.py \
  --dataset chemistry \
  --task nc \
  --res_path ./results/chemistry_nc.jsonl
```

**Metrics reported:** `strict_acc` (exact match) and `overall_acc` (label appears uniquely in answer).

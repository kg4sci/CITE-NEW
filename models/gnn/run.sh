#!/bin/bash
#GPU=${1:-4}
GPU=3
export CUDA_VISIBLE_DEVICES=$GPU

#datasets=("cora" "pubmed" "arxiv" )
#models=("GAT" "SAGE" "GCN")
datasets=("chemistry")
models=("SAGE")
hidden_channels=(16 64 128 256)
num_layers=(2 3)
dropout=(0.3 0.5 0.6)


for model in "${models[@]}"
do
    for dataset in "${datasets[@]}"
    do
        for hidden in "${hidden_channels[@]}"
        do
            for num in "${num_layers[@]}"
            do
                for dr in "${dropout[@]}"
                do
                    python gnn.py --dataname $dataset --model $model --dropout $dr --hidden_channels $hidden --num_layers $num 
                done
            done
        done
    done
done

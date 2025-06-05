export PYTHONPATH=$(pwd)/graphgpt:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2
model_path=./vicuna-7b-v1.5-16k # Vicuna 
instruct_ds=./data/stage_1/stage1.json # dataset
graph_data_path=../datasets/pt/my_graph_data_new.pt # graphdata
pretra_gnn=clip_gt_arxiv   # graphencoder
output_model=./checkpoints/stage_1   # output path


wandb offline

python3 graphgpt/train/train_mem.py \
    --model_name_or_path ${model_path} \
    --version v1 \
    --data_path ${instruct_ds} \
    --graph_content ./arxiv_ti_ab.json \
    --graph_data_path ${graph_data_path} \
    --graph_tower ${pretra_gnn} \
    --tune_graph_mlp_adapter True \
    --graph_select_layer -2 \
    --use_graph_start_end \
    --bf16 True \
    --output_dir ${output_model} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb
	#-m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=20001 
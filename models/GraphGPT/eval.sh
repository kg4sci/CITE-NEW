export PYTHONPATH=$(pwd)/graphgpt:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2
export RAY_TMPDIR=/mnt/data/zch/tmp


output_model=./checkpoints0501/stage_2
datapath=./data/eval/output_hop2_std.json
graph_data_path=./graph_data/my_graph_data_new.pt
res_path=./output_stage_2_my_nc_std
start_id=0
end_id=700
num_gpus=1

python3 ./graphgpt/eval/run_graphgpt.py --model-name ${output_model}  --prompting_file ${datapath} --graph_data_path ${graph_data_path} --output_res_path ${res_path} --start_id ${start_id} --end_id ${end_id} --num_gpus ${num_gpus}
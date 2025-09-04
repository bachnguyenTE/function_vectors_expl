#!/bin/bash
datasets=('antonym')
# datasets=('antonym' 'capitalize' 'country-capital' 'english-french' 'present-past' 'singular-plural')
cd ../

for d_name in "${datasets[@]}"
do
    echo "Running Script for: ${d_name}"
    # python evaluate_function_vector.py --dataset_name="${d_name}" --save_path_root="results/stock_ins" --model_name='meta-llama/Llama-3.1-8B-Instruct' --fv_cot --cot_length=100 --cot_instruct
    # python evaluate_function_vector.py --dataset_name="${d_name}" --save_path_root="results/reasoning-CoT_FV" --model_name='deepseek-ai/DeepSeek-R1-Distill-Llama-8B' --generate_cot --cot_length=100 # --fv_from_cot
    python evaluate_function_vector.py --dataset_name="${d_name}" --save_path_root="results/reasoning-from-stock_ins" --model_name='deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
    # python evaluate_function_vector.py --dataset_name="${d_name}" --save_path_root="results/stock-from-reasoning" --model_name='meta-llama/Llama-3.1-8B'
    # python evaluate_function_vector.py --dataset_name="${d_name}" --save_path_root="results/stock-from-stock_ins" --model_name='meta-llama/Llama-3.1-8B'
    # python evaluate_function_vector.py --dataset_name="${d_name}" --save_path_root="results/stock_ins-from-reasoning" --model_name='meta-llama/Llama-3.1-8B-Instruct'
    # python evaluate_function_vector.py --dataset_name="${d_name}" --save_path_root="results/stock_ins-from-stock" --model_name='meta-llama/Llama-3.1-8B-Instruct'
done
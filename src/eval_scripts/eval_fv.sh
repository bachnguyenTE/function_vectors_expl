#!/bin/bash
datasets=('antonym')
# datasets=('antonym' 'capitalize' 'country-capital' 'english-french' 'present-past' 'singular-plural')
cd ../

for d_name in "${datasets[@]}"
do
    echo "Running Script for: ${d_name}"
    #python evaluate_function_vector.py --dataset_name="${d_name}" --save_path_root="results/stock_ins-CoT" --model_name='meta-llama/Llama-3.1-8B-Instruct' --generate_cot --cot_length=100 --cot_instruct
    python evaluate_function_vector.py --dataset_name="${d_name}" --save_path_root="results/reasoning" --model_name='deepseek-ai/DeepSeek-R1-Distill-Llama-8B' --fv_cot --cot_length=100
done
#! /bin/bash

export SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd $PROJECT_DIR
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"

export llama_tokenizer_path="/home/jiarui/code/LWM/checkpoints/LWM-Text-Chat-1M-Jax/tokenizer.model"
export lwm_text_checkpoint="/home/jiarui/code/LWM/checkpoints/LWM-Text-Chat-1M-Jax/params"

python3 -u scripts/eval_speed.py \
    --mesh_dim='!1,1,-1,1' \
    --dtype='fp32' \
    --load_llama_config='7b' \
    --update_llama_config="dict(theta=50000000,max_sequence_length=1048576,scan_attention=True,scan_query_chunk_size=1024,scan_key_chunk_size=1024,scan_mlp=True,scan_mlp_chunk_size=1024,scan_layers=True)" \
    --load_checkpoint="params::$lwm_text_checkpoint" \
    --tokenizer.vocab_file="$llama_tokenizer_path" \
    --max_tokens_per_batch=1000000 \
    --context_lengths_min=100000 \
    --n_context_length_intervals=10 \
    --context_lengths_max=1000000 \
read
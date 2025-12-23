pyenv local 3.11.9

source src/rai_finetune/setup_finetune_shell.sh

# rm -rf qwen_finetuned
# rm -rf merged_model
# rm -rf qwen_ollama_test
# rm -rf merged_model_16bit

# finetune the model
# python src/rai_finetune/rai_finetune/model/trainer.py \
#   --model "unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit" \
#   --training-data "training_data.jsonl" \
#   --output-dir "qwen_finetuned" \
#   --epochs 1 \
#   --batch-size 1 \
#   --learning-rate 1e-5 \
#   --gradient-accumulation 1 \
#   --max-seq-length 2048 \
#   --max-samples 10 \
#   --data-format "chatml" \
#   --chat-template "src/rai_finetune/rai_finetune/model/tool_aware_template.jinja"

# python -m rai_finetune.model.trainer --training-data ../../training_data_fixed.jsonl --output-dir ../../qwen_finetuned_test 
# --epochs 1 --batch-size 1 --max-samples 10 --chat-template tool_aware_template.jinja

# Use Unsloth's native merging (recommended)
# python src/rai_finetune/rai_finetune/model/merge_with_base_model.py --adapter-dir ./qwen_finetuned --save-method merged_4bit_forced

# # Auto-detect base model from adapter config
# python src/rai_finetune/rai_finetune/model/merge_with_base_model.py --adapter-dir ./qwen_finetuned

# # Specify base model explicitly  
# python src/rai_finetune/rai_finetune/model/merge_with_base_model.py --adapter-dir ./qwen_finetuned --base-model-path unsloth/Qwen2.5-7B-Instruct


# make sure llama.cpp is accessible
# git clone https://github.com/ggml-org/llama.cpp 

OLLAMA_MODEL_PATH="qwen_ollama_test"
OLLAMA_LLAMA_CPP_PATH="llama.cpp"
LLAMA_QUANTIZE_PATH=~/dev/llama.cpp/build/bin/llama-quantize

mkdir -p $OLLAMA_LLAMA_CPP_PATH
cp $LLAMA_QUANTIZE_PATH $OLLAMA_LLAMA_CPP_PATH/llama-quantize



# Ollama model producer
# python src/rai_finetune/rai_finetune/model/ollama_model_producer.py \
#     --input-dir qwen_finetuned \
#     --base-model-path ~/.cache/huggingface/hub/models--unsloth--qwen2.5-7b-instruct/snapshots/a75c9dc945567a9b6f568b8503a0307731607bee \
#     --output-dir qwen_ollama

# python src/rai_finetune/rai_finetune/model/ollama_model_producer.py \
#     --input-dir qwen_finetuned \
#     --base-model-path unsloth/Qwen3-4B-Instruct-2507 \
#     --output-dir qwen_ollama



python src/rai_finetune/rai_finetune/model/ollama_model_producer.py \
       --merged-model-path merged_model \
       --output-dir $OLLAMA_MODEL_PATH      

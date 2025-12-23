source src/rai_finetune/setup_finetune_shell.sh

# run the model
ollama create qwen-finetuned_test  -f qwen_finetuned_ollama_test/Modelfile
# ollama run qwen-finetuned_test
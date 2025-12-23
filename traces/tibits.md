# Tibits learned


Ollama thinking:  
https://ollama.com/blog/thinking


AWS
https://www.amazon.science/blog/enabling-llms-to-make-the-right-api-calls-in-the-right-order

W&B 
https://wandb.ai/wandb/function-calling-finetuning/reports/Fine-tuning-LLMs-for-function-calling--VmlldzoxMjgxMTgxMg
- Nested function calls are fragile because each step depends entirely on the previous one being correct. If the model makes even a small mistake - like passing the wrong format, misnaming a parameter, or misunderstanding an intermediate result - the entire chain breaks. Unlike sequential calls where each function can stand somewhat on its own, nested calls require perfect alignment across multiple steps. One broken link and the whole process fails.
- Good article to revisit when constructing actual dataset


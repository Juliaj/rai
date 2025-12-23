#!/usr/bin/env python3
"""
Very simple test to check if fine-tuned model can generate anything
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def simple_test():
    print("Testing base model without adapter...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-7B-Instruct")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            "unsloth/Qwen2.5-7B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        simple_prompt = "Hello"
        inputs = tokenizer(simple_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        
        print("Generating with base model...")
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Base model works! Generated: {response}")
        return True
        
    except Exception as e:
        print(f"❌ Base model error: {e}")
        return False

if __name__ == "__main__":
    simple_test() 
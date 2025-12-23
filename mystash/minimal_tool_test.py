#!/usr/bin/env python3
"""
Minimal tool calling test with short generation sequences
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def minimal_tool_test():
    print("Loading model for minimal tool calling test...")
    
    tokenizer = AutoTokenizer.from_pretrained("merged_model")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        "merged_model",
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    
    print("✅ Model loaded successfully!")
    
    # Test 1: Very short tool calling prompt
    print("\n" + "="*50)
    print("TEST 1: Short tool calling prompt")
    print("="*50)
    
    short_prompt = """<|im_start|>user
Use get_weather tool.
<|im_end|>
<|im_start|>assistant
<|tool_calls|>"""
    
    print("Input prompt:")
    print(short_prompt)
    print("\nGenerating response (max 10 tokens)...")
    
    inputs = tokenizer(short_prompt, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=10,  # Very short generation
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated response: {response}")
    
    # Test 2: Even shorter prompt
    print("\n" + "="*50)
    print("TEST 2: Minimal tool prompt")
    print("="*50)
    
    minimal_prompt = """<|im_start|>user
Weather tool.
<|im_end|>
<|im_start|>assistant
<|tool_calls|>"""
    
    print("Input prompt:")
    print(minimal_prompt)
    print("\nGenerating response (max 5 tokens)...")
    
    inputs = tokenizer(minimal_prompt, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=5,  # Extremely short
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated response: {response}")
    
    print("\n" + "="*50)
    print("MINIMAL TOOL TESTING COMPLETED")
    print("="*50)
    
    return True

if __name__ == "__main__":
    minimal_tool_test() 
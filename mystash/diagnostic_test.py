#!/usr/bin/env python3
"""
Diagnostic test to understand what causes the hanging
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def diagnostic_test():
    print("Loading model for diagnostic testing...")
    
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
    
    # Test 1: Can it handle tool-related text without generation?
    print("\n" + "="*50)
    print("TEST 1: Tool-related text without generation")
    print("="*50)
    
    tool_text = "The get_weather tool is useful for checking weather."
    
    print("Input text:")
    print(tool_text)
    print("\nTesting if model can process this text...")
    
    inputs = tokenizer(tool_text, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    print("✅ Model processed tool-related text successfully")
    
    # Test 2: Can it generate after seeing tool-related text?
    print("\n" + "="*50)
    print("TEST 2: Generate after tool-related text")
    print("="*50)
    
    prompt = "The get_weather tool is useful. I will"
    
    print("Input prompt:")
    print(prompt)
    print("\nGenerating response (max 3 tokens)...")
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=3,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated response: {response}")
    
    # Test 3: Can it handle the <|tool_calls|> token specifically?
    print("\n" + "="*50)
    print("TEST 3: <|tool_calls|> token handling")
    print("="*50)
    
    tool_calls_prompt = "Here is a tool call: <|tool_calls|>"
    
    print("Input prompt:")
    print(tool_calls_prompt)
    print("\nGenerating response (max 2 tokens)...")
    
    inputs = tokenizer(tool_calls_prompt, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated response: {response}")
    
    print("\n" + "="*50)
    print("DIAGNOSTIC TESTING COMPLETED")
    print("="*50)
    
    return True

if __name__ == "__main__":
    diagnostic_test() 
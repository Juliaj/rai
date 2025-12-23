#!/usr/bin/env python3
"""
Test tool calling with the correct format that matches the training data
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def correct_format_test():
    print("Loading model for correct format testing...")
    
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
    
    # Test with the correct format from training data
    print("\n" + "="*60)
    print("TEST: Correct tool calling format")
    print("="*60)
    
    correct_prompt = """{"messages": [{"role": "user", "content": "get the weather"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\": \"Tokyo\", \"unit\": \"celsius\"}"}}]}]}"""
    
    print("Input prompt (correct format):")
    print(correct_prompt)
    print("\nGenerating response (max 20 tokens)...")
    
    inputs = tokenizer(correct_prompt, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated response: {response}")
    
    # Test with a simpler prompt that matches training data
    print("\n" + "="*60)
    print("TEST: Simple tool request")
    print("="*60)
    
    simple_prompt = """{"messages": [{"role": "user", "content": "take a picture"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "call_456", "type": "function", "function": {"name": "get_ros2_camera_image", "arguments": "{}"}}]}]}"""
    
    print("Input prompt:")
    print(simple_prompt)
    print("\nGenerating response (max 15 tokens)...")
    
    inputs = tokenizer(simple_prompt, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=15,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated response: {response}")
    
    print("\n" + "="*60)
    print("CORRECT FORMAT TESTING COMPLETED")
    print("="*60)
    
    return True

if __name__ == "__main__":
    correct_format_test() 
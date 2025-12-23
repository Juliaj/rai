#!/usr/bin/env python3
"""
Test the new merged model for tool calling functionality
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_new_merged_model():
    print("Testing new merged model for tool calling...")
    
    try:
        # Load the new merged model
        print("Loading merged model...")
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
        
        # Test 1: Simple tool calling prompt
        print("\n" + "="*60)
        print("TEST 1: Simple tool calling")
        print("="*60)
        
        prompt = """{"messages": [{"role": "user", "content": "get the weather in Tokyo"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\": \"Tokyo\", \"unit\": \"celsius\"}"}}]}]}"""
        
        print("Input prompt:")
        print(prompt)
        print("\nGenerating response (max 20 tokens)...")
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
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
        
        # Test 2: Tool calling continuation
        print("\n" + "="*60)
        print("TEST 2: Tool calling continuation")
        print("="*60)
        
        prompt2 = """{"messages": [{"role": "user", "content": "move an object"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "call_456", "type": "function", "function": {"name": "move_object_from_to", "arguments": "{\"x\": 0.1, \"y\": 0.0, \"z\": 0.1, \"x1\": 0.3, \"y1\": 0.0, \"z1\": 0.1}"}}]}]}"""
        
        print("Input prompt:")
        print(prompt2)
        print("\nGenerating response (max 15 tokens)...")
        
        inputs = tokenizer(prompt2, return_tensors="pt", padding=True, truncation=True)
        
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
        print("TESTING COMPLETED")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_new_merged_model()
    if success:
        print("\n🎉 SUCCESS: New merged model is working!")
    else:
        print("\n💥 FAILURE: New merged model has issues.") 
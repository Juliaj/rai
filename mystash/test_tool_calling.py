#!/usr/bin/env python3
"""
Simple smoke test to check if fine-tuned model supports tool calling
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

def test_tool_calling():
    # Load the base model and tokenizer
    base_model_name = "unsloth/Qwen2.5-7B-Instruct"
    adapter_path = "qwen_finetuned"
    
    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Ensure pad token is different from eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"Loading fine-tuned adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Simple tool calling test prompt
    test_prompt = """<|im_start|>system
You are a helpful assistant that can use tools. When you need to use a tool, respond with a JSON object containing the tool name and parameters.
<|im_end|>
<|im_start|>user
What's the weather like in New York? Use the get_weather tool.
<|im_end|>
<|im_start|>assistant
I'll check the weather in New York for you.
<|tool_calls|>
<|invoke|>
<|tool_name|>
get_weather
<|args|>
<|json|>
{"location": "New York", "unit": "celsius"}
<|json|>
<|args|>
<|invoke|>
<|tool_calls|>
<|im_end|>"""
    
    print("\nTesting tool calling with prompt...")
    print("=" * 50)
    
    # Generate response with proper attention mask
    inputs = tokenizer(test_prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Move inputs to device and ensure proper structure
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    try:
        with torch.no_grad():
            # Use fast greedy decoding instead of sampling
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,   # Much smaller for faster testing
                do_sample=False,     # Use greedy decoding (faster)
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("Generated response:")
        print(response)
        print("\n" + "=" * 50)
        
        # Check if response contains tool calling patterns
        tool_indicators = ["<|tool_calls|>", "get_weather", "tool_name", "args"]
        found_indicators = [indicator for indicator in tool_indicators if indicator in response]
        
        print(f"\nTool calling indicators found: {found_indicators}")
        
        if len(found_indicators) >= 2:
            print("✅ Model appears to support tool calling!")
        else:
            print("❌ Model may not support tool calling properly")
        
        return response
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        print("\nTrying to debug the issue...")
        
        # Try to get model info
        try:
            print(f"Model device: {next(model.parameters()).device}")
            print(f"Model dtype: {next(model.parameters()).dtype}")
            print(f"Input shape: {input_ids.shape}")
            print(f"Attention mask shape: {attention_mask.shape}")
        except Exception as debug_e:
            print(f"Debug info error: {debug_e}")
        
        return None

if __name__ == "__main__":
    test_tool_calling() 
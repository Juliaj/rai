#!/usr/bin/env python3
"""
Test tool calling with the merged fine-tuned model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_merged_model():
    print("Loading merged fine-tuned model...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("merged_model")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            "merged_model",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Test tool calling prompt
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
        
        print("Testing tool calling with merged model...")
        print("=" * 50)
        
        inputs = tokenizer(test_prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=100,
                do_sample=False,  # Use greedy decoding for speed
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("Generated response:")
        print(response)
        print("\n" + "=" * 50)
        
        # Check for tool calling patterns
        tool_indicators = ["<|tool_calls|>", "get_weather", "tool_name", "args"]
        found_indicators = [indicator for indicator in tool_indicators if indicator in response]
        
        print(f"\nTool calling indicators found: {found_indicators}")
        
        if len(found_indicators) >= 2:
            print("✅ Merged model supports tool calling!")
        else:
            print("❌ Merged model may not support tool calling properly")
        
        return response
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    test_merged_model() 
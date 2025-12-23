#!/usr/bin/env python3
"""
Test tool calling with the merged fine-tuned model using HuggingFace directly
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_tool_calling_with_merged():
    print("Loading merged fine-tuned model...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("merged_model")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        try:
            print("Attempting to load model with GPU offloading...")
            model = AutoModelForCausalLM.from_pretrained(
                "merged_model",
                torch_dtype=torch.float16,
                device_map="auto",
                llm_int8_enable_fp32_cpu_offload=True,
                low_cpu_mem_usage=True
            )
        except Exception as gpu_error:
            print(f"GPU loading failed: {gpu_error}")
            print("Falling back to CPU-only loading...")
            model = AutoModelForCausalLM.from_pretrained(
                "merged_model",
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
        
        print("✅ Model loaded successfully!")
        
        # Test 1: Simple tool calling prompt
        print("\n" + "="*60)
        print("TEST 1: Simple tool calling prompt")
        print("="*60)
        
        simple_prompt = """<|im_start|>user
Check the weather in Tokyo using the get_weather tool.
<|im_end|>
<|im_start|>assistant
I'll check the weather in Tokyo for you.
<|tool_calls|>
<|invoke|>
<|tool_name|>
get_weather
<|args|>
<|json|>
{"location": "Tokyo", "unit": "celsius"}
<|json|>
<|args|>
<|invoke|>
<|tool_calls|>
<|im_end|>"""
        
        print("Input prompt:")
        print(simple_prompt)
        print("\nGenerating response...")
        
        inputs = tokenizer(simple_prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("\nGenerated response:")
        print(response)
        
        # Test 2: More complex tool calling scenario
        print("\n" + "="*60)
        print("TEST 2: Complex tool calling scenario")
        print("="*60)
        
        complex_prompt = """<|im_start|>system
You are a helpful assistant that can use tools. You have access to tools for object detection, manipulation, and camera operations. When you need to use a tool, respond with the proper tool calling format.
<|im_end|>
<|im_start|>user
I need to move an object from position (10, 5, 2) to position (15, 8, 3). Can you help me with that?
<|im_end|>
<|im_start|>assistant
I'll help you move that object. Let me use the move_object_from_to tool to accomplish this task.
<|tool_calls|>
<|invoke|>
<|tool_name|>
move_object_from_to
<|args|>
<|json|>
{"x": 10, "y": 5, "z": 2, "x1": 15, "y1": 8, "z1": 3}
<|json|>
<|args|>
<|invoke|>
<|tool_calls|>
<|im_end|>"""
        
        print("Input prompt:")
        print(complex_prompt)
        print("\nGenerating response...")
        
        inputs = tokenizer(complex_prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("\nGenerated response:")
        print(response)
        
        # Test 3: Check if model can continue tool calling patterns
        print("\n" + "="*60)
        print("TEST 3: Tool calling continuation")
        print("="*60)
        
        continuation_prompt = """<|im_start|>user
Get me the current camera image.
<|im_end|>
<|im_start|>assistant
I'll capture an image from the camera for you.
<|tool_calls|>
<|invoke|>
<|tool_name|>
get_ros2_camera_image
<|args|>
<|json|>
{}
<|json|>
<|args|>
<|invoke|>
<|tool_calls|>
<|im_end|>"""
        
        print("Input prompt:")
        print(continuation_prompt)
        print("\nGenerating response...")
        
        inputs = tokenizer(continuation_prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("\nGenerated response:")
        print(response)
        
        print("\n" + "="*60)
        print("TESTING COMPLETED")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_tool_calling_with_merged() 
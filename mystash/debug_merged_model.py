#!/usr/bin/env python3
"""
Debug script to isolate where the merged model is hanging
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

def debug_merged_model():
    print("🔍 Debugging merged model step by step...")
    
    try:
        # Step 1: Load tokenizer
        print("\n" + "="*60)
        print("STEP 1: Loading tokenizer...")
        start_time = time.time()
        
        tokenizer = AutoTokenizer.from_pretrained("merged_model")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"✅ Tokenizer loaded in {time.time() - start_time:.2f}s")
        print(f"   Pad token: {tokenizer.pad_token}")
        print(f"   EOS token: {tokenizer.eos_token}")
        
        # Step 2: Load model
        print("\n" + "="*60)
        print("STEP 2: Loading model...")
        start_time = time.time()
        
        model = AutoModelForCausalLM.from_pretrained(
            "merged_model",
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        print(f"✅ Model loaded in {time.time() - start_time:.2f}s")
        print(f"   Model device: {next(model.parameters()).device}")
        print(f"   Model dtype: {next(model.parameters()).dtype}")
        
        # Step 3: Test basic tokenization
        print("\n" + "="*60)
        print("STEP 3: Testing basic tokenization...")
        start_time = time.time()
        
        simple_text = "Hello world"
        inputs = tokenizer(simple_text, return_tensors="pt", padding=True, truncation=True)
        
        print(f"✅ Tokenization successful in {time.time() - start_time:.2f}s")
        print(f"   Input shape: {inputs.input_ids.shape}")
        print(f"   Attention mask shape: {inputs.attention_mask.shape}")
        
        # Step 4: Test basic forward pass
        print("\n" + "="*60)
        print("STEP 4: Testing basic forward pass...")
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✅ Forward pass successful in {time.time() - start_time:.2f}s")
        print(f"   Output shape: {outputs.logits.shape}")
        
        # Step 5: Test minimal generation (1 token)
        print("\n" + "="*60)
        print("STEP 5: Testing minimal generation (1 token)...")
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        print(f"✅ Generation successful in {time.time() - start_time:.2f}s")
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   Generated: {response}")
        
        # Step 6: Test tool calling generation (5 tokens)
        print("\n" + "="*60)
        print("STEP 6: Testing tool calling generation (5 tokens)...")
        start_time = time.time()
        
        tool_prompt = """{"messages": [{"role": "user", "content": "get weather"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\": \"Tokyo\"}"}}]}]}"""
        
        tool_inputs = tokenizer(tool_prompt, return_tensors="pt", padding=True, truncation=True)
        
        print("   Tool prompt tokenized successfully")
        print(f"   Tool input shape: {tool_inputs.input_ids.shape}")
        
        with torch.no_grad():
            tool_outputs = model.generate(
                tool_inputs.input_ids,
                attention_mask=tool_inputs.attention_mask,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        print(f"✅ Tool generation successful in {time.time() - start_time:.2f}s")
        tool_response = tokenizer.decode(tool_outputs[0], skip_special_tokens=True)
        print(f"   Generated: {tool_response}")
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED! Model is working correctly.")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error at step: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_merged_model()
    if not success:
        print("\n💥 DEBUGGING FAILED - Check the error above.") 
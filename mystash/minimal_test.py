#!/usr/bin/env python3
"""
Minimal test to isolate the hanging issue
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def minimal_test():
    print("Step 1: Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("merged_model")
        print("✅ Tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ Tokenizer error: {e}")
        return False
    
    print("\nStep 2: Loading model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "merged_model",
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False
    
    print("\nStep 3: Testing basic tokenization...")
    try:
        test_text = "Hello"
        inputs = tokenizer(test_text, return_tensors="pt")
        print(f"✅ Tokenization successful: {inputs.input_ids.shape}")
    except Exception as e:
        print(f"❌ Tokenization error: {e}")
        return False
    
    print("\nStep 4: Testing basic forward pass...")
    try:
        with torch.no_grad():
            outputs = model(**inputs)
        print("✅ Forward pass successful")
    except Exception as e:
        print(f"❌ Forward pass error: {e}")
        return False
    
    print("\nStep 5: Testing minimal generation...")
    try:
        # Fix attention mask issue
        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=1,  # Generate just 1 token
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Generation successful: {response}")
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return False
    
    print("\n✅ All tests passed! The model is working.")
    return True

if __name__ == "__main__":
    print("Starting minimal test...")
    result = minimal_test()
    if result:
        print("\n🎉 SUCCESS: Your fine-tuned model is working!")
    else:
        print("\n💥 FAILURE: There's a fundamental issue with the model.") 
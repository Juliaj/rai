#!/usr/bin/env python3

"""
Test script to validate the improved LoRA merger functionality
"""

import json
import tempfile
from pathlib import Path
from rai_finetune.model.merge_with_base_model import LoRAMerger

def create_test_adapter_config(temp_dir: Path, base_model_name: str):
    """Create a minimal adapter config for testing"""
    # Ensure directory exists
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    adapter_config = {
        "base_model_name_or_path": base_model_name,
        "peft_type": "LORA",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj"]
    }
    
    config_path = temp_dir / "adapter_config.json"
    with open(config_path, "w") as f:
        json.dump(adapter_config, f, indent=2)
    
    # Create dummy adapter model file
    adapter_model_path = temp_dir / "adapter_model.safetensors"
    adapter_model_path.touch()
    
    return temp_dir

def test_auto_detection():
    """Test automatic base model detection from adapter config"""
    print("Testing automatic base model detection...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test adapter config
        adapter_dir = create_test_adapter_config(temp_path / "adapter", "microsoft/DialoGPT-medium")
        
        try:
            # This should attempt to auto-detect the base model
            merger = LoRAMerger(str(adapter_dir), base_model_path=None)
            print(f"✅ Auto-detection successful. Resolved path: {merger.base_model_path}")
            print(f"   Base model ID: {merger._base_model_id}")
        except Exception as e:
            print(f"⚠️  Auto-detection failed (expected if model not cached): {e}")

def test_huggingface_id():
    """Test handling of HuggingFace model IDs"""
    print("\nTesting HuggingFace model ID handling...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test adapter config
        adapter_dir = create_test_adapter_config(temp_path / "adapter", "microsoft/DialoGPT-medium")
        
        try:
            # Pass a HuggingFace model ID that doesn't exist locally
            merger = LoRAMerger(str(adapter_dir), base_model_path="microsoft/DialoGPT-medium")
            print(f"✅ HuggingFace ID handling successful. Resolved path: {merger.base_model_path}")
            print(f"   Base model ID: {merger._base_model_id}")
        except Exception as e:
            print(f"⚠️  HuggingFace ID handling failed (expected if model not available): {e}")

def test_local_path():
    """Test handling of local paths"""
    print("\nTesting local path handling...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a mock base model directory
        base_model_dir = temp_path / "base_model"
        base_model_dir.mkdir()
        
        # Create required files
        (base_model_dir / "config.json").write_text('{"model_type": "gpt2"}')
        (base_model_dir / "pytorch_model.bin").touch()
        (base_model_dir / "tokenizer.json").touch()
        
        # Create test adapter config
        adapter_dir = create_test_adapter_config(temp_path / "adapter", str(base_model_dir))
        
        try:
            merger = LoRAMerger(str(adapter_dir), base_model_path=str(base_model_dir))
            print(f"✅ Local path handling successful. Resolved path: {merger.base_model_path}")
            print(f"   Base model ID: {merger._base_model_id}")
            
            # Test validation
            is_valid = merger._validate_base_model_path()
            print(f"   Validation result: {is_valid}")
            
        except Exception as e:
            print(f"❌ Local path handling failed: {e}")

if __name__ == "__main__":
    print("Testing improved LoRA merger functionality")
    print("=" * 50)
    
    test_auto_detection()
    test_huggingface_id()
    test_local_path()
    
    print("\n" + "=" * 50)
    print("Test completed!") 
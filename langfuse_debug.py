#!/usr/bin/env python3
"""
Langfuse Debugging Script for RAI
This script helps debug Langfuse tracing issues by testing various components.
"""

import os
import logging
import sys
from uuid import uuid4

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_environment_variables():
    """Test if environment variables are properly set."""
    print("=" * 50)
    print("1. TESTING ENVIRONMENT VARIABLES")
    print("=" * 50)
    
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    
    print(f"LANGFUSE_PUBLIC_KEY: pk-lf-f058bac2-7d1b-4b79-a0cd-bf480619f6f0 {'✓' if public_key else '✗'} {public_key[:20]}..." if public_key else "Not set")
    print(f"LANGFUSE_SECRET_KEY: sk-lf-b21159a7-d345-4e2c-80c3-54480347caa0 {'✓' if secret_key else '✗'} {secret_key[:20]}..." if secret_key else "Not set")
    
    return public_key is not None and secret_key is not None

def test_config_loading():
    """Test RAI config loading."""
    print("\n" + "=" * 50)
    print("2. TESTING CONFIG LOADING")
    print("=" * 50)
    
    try:
        from rai.initialization.model_initialization import load_config
        config = load_config()
        
        print(f"Langfuse enabled: {config.tracing.langfuse.use_langfuse}")
        print(f"Langfuse host: {config.tracing.langfuse.host}")
        print(f"Project name: {config.tracing.project}")
        
        return config.tracing.langfuse.use_langfuse
    except Exception as e:
        print(f"Error loading config: {e}")
        return False

def test_langfuse_direct():
    """Test Langfuse directly without RAI."""
    print("\n" + "=" * 50)
    print("3. TESTING LANGFUSE DIRECT CONNECTION")
    print("=" * 50)
    
    try:
        from langfuse import Langfuse
        
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = "http://localhost:3000"
        
        langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host
        )
        
        # Test creating a trace directly
        trace = langfuse.trace(
            name="debug_test_trace",
            metadata={"test": "direct_langfuse"}
        )
        
        print(f"✓ Direct trace created with ID: {trace.id}")
        
        # Add a generation to the trace
        generation = trace.generation(
            name="debug_generation",
            model="test-model",
            input="Hello world",
            output="Hi there!"
        )
        
        print(f"✓ Generation created with ID: {generation.id}")
        
        # Flush to ensure data is sent
        langfuse.flush()
        print("✓ Langfuse flush completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Error with direct Langfuse: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rai_callbacks():
    """Test RAI's get_tracing_callbacks function."""
    print("\n" + "=" * 50)
    print("4. TESTING RAI TRACING CALLBACKS")
    print("=" * 50)
    
    try:
        from rai.initialization import get_tracing_callbacks
        
        callbacks = get_tracing_callbacks()
        print(f"Number of callbacks: {len(callbacks)}")
        
        for i, callback in enumerate(callbacks):
            print(f"Callback {i}: {type(callback).__name__}")
            
            # Check if it's a Langfuse callback
            if hasattr(callback, 'langfuse'):
                print(f"  - Has langfuse attribute: ✓")
                print(f"  - Langfuse client: {callback.langfuse}")
            else:
                print(f"  - No langfuse attribute: ✗")
        
        return len(callbacks) > 0
        
    except Exception as e:
        print(f"✗ Error getting tracing callbacks: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_langfuse_callback():
    """Test Langfuse callback directly."""
    print("\n" + "=" * 50)
    print("5. TESTING LANGFUSE CALLBACK HANDLER")
    print("=" * 50)
    
    try:
        from langfuse.callback import CallbackHandler
        
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = "http://localhost:3000"
        
        callback = CallbackHandler(
            public_key=public_key,
            secret_key=secret_key,
            host=host
        )
        
        print(f"✓ Callback handler created: {callback}")
        print(f"✓ Langfuse client: {callback.langfuse}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating callback handler: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_llm_with_tracing():
    """Test a simple LLM call with Langfuse tracing."""
    print("\n" + "=" * 50)
    print("6. TESTING LLM WITH LANGFUSE TRACING")
    print("=" * 50)
    
    try:
        from rai.initialization import get_tracing_callbacks, get_llm_model
        from langchain_core.messages import HumanMessage
        
        # Get LLM
        llm = get_llm_model("simple_model")
        print(f"✓ LLM loaded: {type(llm).__name__}")
        
        # Get callbacks
        callbacks = get_tracing_callbacks()
        print(f"✓ Got {len(callbacks)} callbacks")
        
        # Test simple invoke
        print("Invoking LLM with tracing...")
        result = llm.invoke(
            [HumanMessage(content="Say 'Hello World' and nothing else.")],
            config={"callbacks": callbacks}
        )
        
        print(f"✓ LLM response: {result.content}")
        
        # Flush Langfuse
        for callback in callbacks:
            if hasattr(callback, 'langfuse'):
                callback.langfuse.flush()
                print("✓ Langfuse flushed")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing LLM with tracing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_langfuse_server_health():
    """Test Langfuse server health endpoint."""
    print("\n" + "=" * 50)
    print("7. TESTING LANGFUSE SERVER HEALTH")
    print("=" * 50)
    
    try:
        import requests
        
        response = requests.get("http://localhost:3000/api/health", timeout=5)
        print(f"Health endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            print("✓ Langfuse server is healthy")
            return True
        else:
            print(f"✗ Langfuse server returned status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Error connecting to Langfuse server: {e}")
        return False
    except ImportError:
        print("✗ requests library not available, skipping server health check")
        return False

def main():
    """Run all debugging tests."""
    print("LANGFUSE DEBUGGING SCRIPT")
    print("========================")
    
    results = []
    
    # Run all tests
    results.append(("Environment Variables", test_environment_variables()))
    results.append(("Config Loading", test_config_loading()))
    results.append(("Langfuse Server Health", test_langfuse_server_health()))
    results.append(("Langfuse Direct", test_langfuse_direct()))
    results.append(("RAI Callbacks", test_rai_callbacks()))
    results.append(("Langfuse Callback", test_langfuse_callback()))
    results.append(("LLM with Tracing", test_simple_llm_with_tracing()))
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed < total:
        print("\nSome tests failed. Check the output above for details.")
        print("Common issues:")
        print("- Langfuse server not running: docker run -p 3000:3000 langfuse/langfuse")
        print("- Network issues: Check firewall/docker networking")
        print("- API key issues: Verify keys in Langfuse UI")
        sys.exit(1)
    else:
        print("\n✓ All tests passed! Langfuse should be working correctly.")

if __name__ == "__main__":
    main() 

# Lost of tool calling

## Issue
works with qwen2.5:7b-instruct and fails with qwen-finetuned  qwen_finetuned uses quen2.5:7b-instruct as based model and fine tuned with collected data. The fine tune code can be found at rai_finetune folder. the error for qwen-finetuned indicates loss of tool calling support. Review the implementation and come up with hypothesis on why this happened. 
  File "/home/juliaj/.cache/pypoetry/virtualenvs/rai-framework--61gjQXo-py3.12/lib/python3.12/site-packages/langchain_core/language_models/chat_models.py", line 1028, in _generate_with_cache
    result = self._generate(
             ^^^^^^^^^^^^^^^
  File "/home/juliaj/.cache/pypoetry/virtualenvs/rai-framework--61gjQXo-py3.12/lib/python3.12/site-packages/langchain_ollama/chat_models.py", line 800, in _generate
    final_chunk = self._chat_stream_with_aggregation(
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/juliaj/.cache/pypoetry/virtualenvs/rai-framework--61gjQXo-py3.12/lib/python3.12/site-packages/langchain_ollama/chat_models.py", line 735, in _chat_stream_with_aggregation
    for chunk in self._iterate_over_stream(messages, stop, **kwargs):
  File "/home/juliaj/.cache/pypoetry/virtualenvs/rai-framework--61gjQXo-py3.12/lib/python3.12/site-packages/langchain_ollama/chat_models.py", line 822, in _iterate_over_stream
    for stream_resp in self._create_chat_stream(messages, stop, **kwargs):
  File "/home/juliaj/.cache/pypoetry/virtualenvs/rai-framework--61gjQXo-py3.12/lib/python3.12/site-packages/langchain_ollama/chat_models.py", line 721, in _create_chat_stream
    yield from self._client.chat(**chat_params)
  File "/home/juliaj/.cache/pypoetry/virtualenvs/rai-framework--61gjQXo-py3.12/lib/python3.12/site-packages/ollama/_client.py", line 170, in inner
    raise ResponseError(e.response.text, e.response.status_code) from None
ollama._types.ResponseError: registry.ollama.ai/library/qwen-finetuned:latest does not support tools (status code: 400)
During task with name 'thinker' and id '72d23785-7db0-d716-f125-4a875d2583c5'

## Hypothesis: Why Tool Calling Support Was Lost

1. Training Data Format Issue - Root Cause
The most critical issue is in the training data formatting. Looking at training_data.jsonl, I can see that:
No structured tool calls: The training data contains 0 instances of proper "tool_calls" format
Natural language conversion: Tool calls are converted to natural language like "I will call get_object_positions(object_name=cube)" instead of maintaining the structured JSON format that Ollama expects
Missing function definitions: The training data lacks the formal function schema definitions that enable Ollama to recognize available tools


2. Training Data Formatter Behavior
In src/rai_finetune/rai_finetune/data/training_data_formatter.py (lines 590-624), the format_for_unsloth method deliberately converts tool calls to natural language:
Apply to gpt4o_observ...
This teaches the model to generate natural language descriptions of tool calls rather than the structured format Ollama requires.

3. Missing Model Configuration
The Ollama conversion process (ollama_model_producer.py) creates a basic Modelfile without tool support:
No tools specification: The generated Modelfile lacks tool definitions
Generic system prompt: Uses a generic assistant prompt instead of tool-aware instructions
Missing tool metadata: No information about available functions is preserved during conversion

4. Configuration Preservation Issues
The fine-tuning and conversion pipeline doesn't preserve the original model's tool calling capabilities:
Chat template preservation: While the ChatML template is used, it doesn't include tool-specific formatting
Model metadata loss: Tool calling capabilities aren't explicitly marked in the final model configuration
Function schema absence: No mechanism to embed function definitions in the converted model

5. Ollama Tool Support Requirements
Ollama requires specific model configuration for tool support:
Function definitions: Models must be configured with available tools
Structured output format: Models must generate JSON-formatted tool calls
Tool-aware templates: Chat templates must handle tool calling scenarios
Solutions to Fix Tool Calling Support

## Recommended Fixes:
Preserve Structured Tool Call Format: Modify the training data formatter to maintain JSON tool call structure instead of converting to natural language
Add Tool Definitions to Modelfile: Update the Ollama converter to include tool definitions in the generated Modelfile
Use Tool-Aware Chat Template: Implement a chat template that properly handles tool calling scenarios
Preserve Original Model Capabilities: Ensure the fine-tuning process doesn't override the base model's tool calling infrastructure
Test with Structured Training Data: Create training examples that maintain the proper tool calling format expected by Ollama
The core issue is that the fine-tuning process trained the model to output natural language descriptions of tool calls rather than the structured JSON format that Ollama's tool calling system requires. This fundamental mismatch between training format and runtime expectations is why the fine-tuned model fails with the "does not support tools" error.

## Debug and test

# verify data
head -50 training_data.jsonl | grep -o '"tool_calls"' | wc -l

# after fine tuning adapter is made
python test_tool_calling.py

# merge model

# make ollama compatible version



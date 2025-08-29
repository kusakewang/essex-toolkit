# OpenAI Embeddings Batcher: Enhanced Compatibility

This document describes the enhanced compatibility of the OpenAIEmbeddingBatcher with both SDK and REST API implementations.

## Summary

The `OpenAIEmbeddingBatcher` has been updated to work seamlessly with both OpenAI embeddings LLM implementations:

1. **OpenAIEmbeddingsLLMImpl** - OpenAI SDK-based implementation
2. **OpenAIEmbeddingsRestLLMImpl** - Direct REST API implementation

## Key Changes Made

### 1. Enhanced Type Documentation

Updated the class documentation to explicitly state compatibility:

```python
class OpenAIEmbeddingBatcher(Batcher[EmbeddingInput, EmbeddingOutput]):
    """
    A utility class to batch embeddings using OpenAI's API.

    This batcher works with both OpenAI SDK-based and REST API-based embeddings LLMs:
    - OpenAIEmbeddingsLLMImpl (uses OpenAI Python SDK)
    - OpenAIEmbeddingsRestLLMImpl (uses direct HTTP calls via httpx)
    """
```

### 2. Type Imports for Clarity

Added explicit imports for both implementations:

```python
if TYPE_CHECKING:
    from fnllm.openai.llm.openai_embeddings_llm import OpenAIEmbeddingsLLMImpl
    from fnllm.openai.llm.openai_embeddings_rest_llm import OpenAIEmbeddingsRestLLMImpl
    # ...

OpenAIEmbeddingsLLMType = Union[OpenAIEmbeddingsLLMImpl, OpenAIEmbeddingsRestLLMImpl]
```

### 3. No Code Changes Required

**Importantly, no actual code changes were needed** because:

- Both implementations inherit from `BaseLLM` with identical type parameters
- Both satisfy the `OpenAIEmbeddingsLLM` type alias
- Both implement the same interface expected by the batcher

## Compatibility Details

### Interface Compatibility

Both LLM implementations provide:
- Same input/output types (`OpenAIEmbeddingsInput`, `OpenAIEmbeddingsOutput`)
- Same method signatures (`child()`, `is_reasoning_model()`, etc.)
- Same async call interface (`await llm(text)`)
- Same parameter handling (`OpenAIEmbeddingsParameters`)

### Type System Compatibility

```python
# Both of these work identically with the batcher
OpenAIEmbeddingsLLM = LLM[
    OpenAIEmbeddingsInput,
    OpenAIEmbeddingsOutput,
    None,
    OpenAIEmbeddingsParameters
]
```

### Runtime Compatibility

The batcher uses the LLM through its public interface:
- `await self._llm([texts])` - for batch requests
- `self._llm.child(name)` - for creating child instances
- `self._llm.is_reasoning_model()` - for model type detection

## Usage Examples

### With SDK Implementation

```python
from fnllm.openai.config import PublicOpenAIConfig
from fnllm.openai.factories import create_openai_embeddings_llm

config = PublicOpenAIConfig(api_key=api_key, model=model)
sdk_llm = create_openai_embeddings_llm(config)

batcher = OpenAIEmbeddingBatcher(
    llm=sdk_llm,
    text_service=text_service,
    max_batch_size=4,
    max_batch_tokens=8192,
)
```

### With REST Implementation

```python
from fnllm.openai.factories import create_openai_embeddings_rest_llm

rest_llm = create_openai_embeddings_rest_llm(
    base_url="https://api.openai.com",
    api_key=api_key,
    model=model
)

batcher = OpenAIEmbeddingBatcher(
    llm=rest_llm,  # Same interface!
    text_service=text_service,
    max_batch_size=4,
    max_batch_tokens=8192,
)
```

## Benefits

### 1. Transparent Compatibility
Users can switch between SDK and REST implementations without changing batcher code.

### 2. Consistent Behavior
Both implementations provide identical batching behavior:
- Same batch size limits
- Same token counting
- Same text splitting for large inputs
- Same weighted averaging for split texts

### 3. Error Handling
Both implementations handle errors gracefully within the batcher context.

### 4. Performance Characteristics
Both benefit equally from the batcher's optimizations:
- Reduced API calls through batching
- Automatic text splitting for oversized inputs
- Concurrent processing of multiple texts

## Testing

Comprehensive tests verify compatibility:

1. **Interface tests** - Both implementations satisfy the expected interface
2. **Runtime tests** - Batcher works correctly with both implementations
3. **Type tests** - Type system accepts both implementations
4. **Behavior tests** - Both produce equivalent results

## Migration Path

For existing users:
- **No changes required** - existing code continues to work
- **Optional migration** - can switch to REST implementation if desired
- **Mix and match** - different parts of application can use different implementations

## Conclusion

The OpenAIEmbeddingBatcher now explicitly documents and tests its compatibility with both OpenAI embeddings implementations, providing users with flexibility in their choice of underlying HTTP client while maintaining consistent batching behavior.

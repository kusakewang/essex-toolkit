# OpenAI Embeddings LLM: SDK vs REST API Implementation

This document explains the new REST API implementation for OpenAI Embeddings LLM that works alongside the existing OpenAI SDK implementation.

## Overview

The fnllm library now supports two approaches for generating embeddings using OpenAI's API:

1. **OpenAI SDK Method** (`OpenAIEmbeddingsLLMImpl`) - Uses the official OpenAI Python SDK
2. **REST API Method** (`OpenAIEmbeddingsRestLLMImpl`) - Makes direct HTTP requests using httpx

## Files Added/Modified

### New Files
- `fnllm/openai/llm/openai_embeddings_rest_llm.py` - REST API implementation
- `examples/embeddings_rest_vs_sdk.py` - Usage example comparing both methods
- `notebooks/sample_embeddings_rest_vs_sdk.ipynb` - Interactive notebook demo
- `fnllm_tests/unit/openai/test_openai_embeddings_rest_llm.py` - Unit tests

### Modified Files
- `fnllm/openai/factories/embeddings.py` - Added `create_openai_embeddings_rest_llm()` factory
- `fnllm/openai/llm/__init__.py` - Exports for new class
- `fnllm/openai/factories/__init__.py` - Exports for new factory function

## Usage Examples

### Using the OpenAI SDK (Existing)

```python
from fnllm.openai.config import PublicOpenAIConfig
from fnllm.openai.factories import create_openai_embeddings_llm

config = PublicOpenAIConfig(
    api_key="your-api-key",
    base_url="https://api.openai.com",
    model="text-embedding-3-small"
)

llm = create_openai_embeddings_llm(config)
result = await llm("Hello, world!")
embeddings = result.output.embeddings[0]  # List of floats
```

### Using REST API (New)

```python
from fnllm.openai.factories import create_openai_embeddings_rest_llm

llm = create_openai_embeddings_rest_llm(
    base_url="https://api.openai.com",
    api_key="your-api-key",
    model="text-embedding-3-small"
)

async with llm:
    result = await llm("Hello, world!")
    embeddings = result.output.embeddings[0]  # List of floats
```

### Azure OpenAI Support

Both implementations support Azure OpenAI:

```python
# SDK version
config = AzureOpenAIConfig(
    api_key="your-api-key",
    endpoint="https://your-resource.openai.azure.com",
    deployment="your-deployment-name",
    api_version="2024-02-01"
)

# REST version
llm = create_openai_embeddings_rest_llm(
    base_url="https://your-resource.openai.azure.com",
    api_key="your-api-key",
    model="your-deployment-name",
    api_version="2024-02-01"  # This enables Azure OpenAI format
)
```

## Key Features

### OpenAIEmbeddingsRestLLMImpl

- **Direct HTTP Control**: Uses httpx for fine-grained control over HTTP requests
- **Async Context Manager**: Properly manages HTTP client lifecycle with `async with`
- **Error Handling**: Comprehensive HTTP error handling with detailed error messages
- **Azure OpenAI Support**: Automatic URL and request body formatting for Azure deployments
- **Compatible Interface**: Same interface as SDK version, making them interchangeable

### Factory Function

- **Consistent API**: `create_openai_embeddings_rest_llm()` follows the same pattern as existing factories
- **Full Service Stack**: Supports caching, rate limiting, retries, and other fnllm services
- **Flexible Configuration**: Supports both OpenAI and Azure OpenAI configurations

## When to Use Each Method

### Use OpenAI SDK Method When:
- ✅ You want full OpenAI SDK features and error handling
- ✅ You prefer automatic retries and rate limiting built into the SDK
- ✅ You need maximum type safety with OpenAI's response objects
- ✅ You're building a standard production application

### Use REST API Method When:
- ✅ You need direct HTTP control and customization
- ✅ You want a smaller dependency footprint (just httpx vs full OpenAI SDK)
- ✅ You're working with OpenAI-compatible APIs that may not work with the SDK
- ✅ You need custom authentication or header manipulation
- ✅ You're in environments where the OpenAI SDK isn't available or compatible

## Implementation Details

### URL Construction
- **OpenAI**: `https://api.openai.com/v1/embeddings`
- **Azure OpenAI**: `https://{resource}.openai.azure.com/openai/deployments/{model}/embeddings?api-version={version}`

### Request Body Handling
- Automatically removes `None` values
- For Azure deployments, removes the `model` parameter (since it's in the URL)
- Supports all OpenAI embeddings parameters (dimensions, encoding_format, user, timeout)

### Response Processing
- Parses OpenAI API response format
- Extracts embeddings and usage statistics
- Creates compatible `OpenAIEmbeddingsOutput` objects
- Preserves headers for debugging/logging

### Error Handling
- HTTP status errors with detailed error messages
- JSON parsing errors
- Network connectivity issues
- Rate limiting and quota errors

## Testing

The implementation includes comprehensive unit tests covering:
- URL construction for both OpenAI and Azure
- Request body preparation
- Successful API responses
- HTTP error handling
- Child LLM creation
- Async context manager behavior

Run tests with:
```bash
pytest fnllm_tests/unit/openai/test_openai_embeddings_rest_llm.py -v
```

## Dependencies

The REST API implementation only requires:
- `httpx>=0.27.0` (already in fnllm dependencies)
- `pydantic>=2.8.2` (already in fnllm dependencies)

No additional dependencies are needed beyond what fnllm already requires.

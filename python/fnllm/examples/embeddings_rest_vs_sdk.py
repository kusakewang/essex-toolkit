#!/usr/bin/env python3
"""
Example demonstrating OpenAI Embeddings LLM with both SDK and REST API approaches.

This script shows how to use the fnllm library with both the OpenAI SDK
and direct REST API calls for generating embeddings.
"""

import asyncio
import os
from typing import List

from fnllm.openai.config import PublicOpenAIConfig
from fnllm.openai.factories import (create_openai_embeddings_llm,
                                    create_openai_embeddings_rest_llm)


async def main():
    """Main function demonstrating both embedding approaches."""

    # Configuration
    api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    base_url = "https://api.openai.com"
    model = "text-embedding-3-small"

    # Sample texts to embed
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a powerful programming language.",
        "Machine learning is transforming industries.",
    ]

    print("OpenAI Embeddings LLM Comparison: SDK vs REST API")
    print("=" * 60)

    # Method 1: OpenAI SDK
    print("\n🔧 Method 1: Using OpenAI SDK")
    print("-" * 30)

    config = PublicOpenAIConfig(
        api_key=api_key,
        base_url=base_url,
        model=model
    )

    sdk_llm = create_openai_embeddings_llm(config)

    for i, text in enumerate(texts[:1]):  # Just test first text for demo
        result = await sdk_llm(text)
        embeddings_output = result.output  # Access the actual OpenAIEmbeddingsOutput
        print(f"Text: {text}")
        if embeddings_output and embeddings_output.embeddings:
            print(f"Embedding dimensions: {len(embeddings_output.embeddings[0])}")
            print(f"First 5 values: {embeddings_output.embeddings[0][:5]}")
        print(f"Usage: {embeddings_output.usage if embeddings_output else 'N/A'}")

    # Method 2: REST API
    print("\n🌐 Method 2: Using REST API")
    print("-" * 30)

    rest_llm = create_openai_embeddings_rest_llm(
        base_url=base_url,
        api_key=api_key,
        model=model
    )

    # Use async context manager for proper cleanup
    async with rest_llm:
        for i, text in enumerate(texts[:1]):  # Just test first text for demo
            result = await rest_llm(text)
            embeddings_output = result.output  # Access the actual OpenAIEmbeddingsOutput
            print(f"Text: {text}")
            if embeddings_output and embeddings_output.embeddings:
                print(f"Embedding dimensions: {len(embeddings_output.embeddings[0])}")
                print(f"First 5 values: {embeddings_output.embeddings[0][:5]}")
            print(f"Usage: {embeddings_output.usage if embeddings_output else 'N/A'}")

    print("\n✅ Both methods completed successfully!")

    # Comparison with Azure OpenAI example
    print("\n📋 Azure OpenAI Configuration Example:")
    print("-" * 40)
    print("""
    # For Azure OpenAI, you would use:
    rest_llm = create_openai_embeddings_rest_llm(
        base_url="https://your-resource.openai.azure.com",
        api_key="your-azure-api-key",
        model="your-deployment-name",
        api_version="2024-02-01"  # This enables Azure OpenAI format
    )
    """)

    print("\n🎯 Key Differences:")
    print("-" * 20)
    print("SDK Method:")
    print("  • Full OpenAI SDK features and error handling")
    print("  • Automatic retries and rate limiting")
    print("  • Requires openai package dependency")
    print()
    print("REST API Method:")
    print("  • Direct HTTP control with httpx")
    print("  • Smaller dependency footprint")
    print("  • Works with any OpenAI-compatible API")
    print("  • More flexibility for custom headers/auth")


if __name__ == "__main__":
    # Check if API key is set
    if os.getenv("OPENAI_API_KEY") is None:
        print("⚠️  Please set your OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        exit(1)

    asyncio.run(main())

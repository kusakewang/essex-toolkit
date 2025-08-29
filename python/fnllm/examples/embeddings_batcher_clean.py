#!/usr/bin/env python3
"""
Example demonstrating OpenAI Embeddings Batcher with both SDK and REST API approaches.

This script shows how to use the fnllm batching functionality with both the OpenAI SDK
and direct REST API calls for generating embeddings efficiently.
"""

import asyncio
import os

import tiktoken

from fnllm.openai.config import PublicOpenAIConfig
from fnllm.openai.factories import (create_openai_embeddings_llm,
                                    create_openai_embeddings_rest_llm)
from fnllm.openai.llm.openai_embeddings_batcher import OpenAIEmbeddingBatcher
from fnllm.openai.services.openai_text_service import OpenAITextService


async def main():
    """Main function demonstrating both batching approaches."""

    # Configuration
    api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    base_url = "https://api.openai.com"
    model = "text-embedding-3-small"

    # Sample texts to embed (more texts to demonstrate batching)
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a powerful programming language.",
        "Machine learning is transforming industries.",
        "Natural language processing enables computers to understand human language.",
        "Large language models have revolutionized AI applications.",
        "Vector embeddings capture semantic meaning of text.",
        "Batch processing improves API efficiency and reduces costs.",
        "OpenAI provides powerful embedding models for various applications.",
    ]

    print("OpenAI Embeddings Batcher Comparison: SDK vs REST API")
    print("=" * 65)

    # Create text service for token counting
    text_service = OpenAITextService(tiktoken.get_encoding("cl100k_base"))

    # Batch configuration
    max_batch_size = 4  # Process 4 texts at a time
    max_batch_tokens = 8192  # Token limit per batch

    # Method 1: SDK-based LLM with Batcher
    print("\n🔧 Method 1: Using OpenAI SDK with Batcher")
    print("-" * 45)

    config = PublicOpenAIConfig(
        api_key=api_key,
        base_url=base_url,
        model=model
    )

    sdk_llm = create_openai_embeddings_llm(config)
    sdk_batcher = OpenAIEmbeddingBatcher(
        llm=sdk_llm,
        text_service=text_service,
        max_batch_size=max_batch_size,
        max_batch_tokens=max_batch_tokens,
    )

    print(f"Processing {len(texts)} texts with SDK batcher...")
    # Use the batcher by calling individual texts and collecting futures
    futures = [sdk_batcher(text) for text in texts]
    sdk_results = await asyncio.gather(*futures)

    print(f"✅ SDK Batcher completed successfully!")
    print(f"   Processed {len(texts)} texts")
    print(f"   Generated {len(sdk_results)} embeddings")
    print(f"   First embedding dimensions: {len(sdk_results[0])}")
    print(f"   Sample values: {sdk_results[0][:3]}")

    # Method 2: REST API-based LLM with Batcher
    print("\n🌐 Method 2: Using REST API with Batcher")
    print("-" * 45)

    rest_llm = create_openai_embeddings_rest_llm(
        base_url=base_url,
        api_key=api_key,
        model=model
    )

    # Use async context manager for proper cleanup
    async with rest_llm:
        rest_batcher = OpenAIEmbeddingBatcher(
            llm=rest_llm,
            text_service=text_service,
            max_batch_size=max_batch_size,
            max_batch_tokens=max_batch_tokens,
        )

        print(f"Processing {len(texts)} texts with REST batcher...")
        # Use the batcher by calling individual texts and collecting futures
        futures = [rest_batcher(text) for text in texts]
        rest_results = await asyncio.gather(*futures)

        print(f"✅ REST Batcher completed successfully!")
        print(f"   Processed {len(texts)} texts")
        print(f"   Generated {len(rest_results)} embeddings")
        print(f"   First embedding dimensions: {len(rest_results[0])}")
        print(f"   Sample values: {rest_results[0][:3]}")

    # Compare results
    print("\n📊 Comparison")
    print("-" * 20)

    if len(sdk_results) == len(rest_results):
        # Calculate similarity between first embeddings
        import numpy as np

        sdk_vec = np.array(sdk_results[0])
        rest_vec = np.array(rest_results[0])

        # Cosine similarity
        similarity = np.dot(sdk_vec, rest_vec) / (np.linalg.norm(sdk_vec) * np.linalg.norm(rest_vec))
        print(f"First embedding similarity: {similarity:.6f}")

        if similarity > 0.999:
            print("✅ Both batchers produce nearly identical results!")
        else:
            print("⚠️ Results differ significantly")

    # Demonstrate batching efficiency
    print(f"\n🚀 Batching Benefits:")
    print(f"   • Processed {len(texts)} texts efficiently")
    print(f"   • Batch size limit: {max_batch_size} texts per API call")
    print(f"   • Token limit: {max_batch_tokens} tokens per batch")
    print(f"   • Both implementations benefit from the same batching logic")

    # Show child batcher creation
    print(f"\n🔗 Child Batcher Example:")
    print("-" * 25)

    child_batcher = sdk_batcher.child("child-batcher")
    print(f"   Created child batcher: {type(child_batcher).__name__}")
    print(f"   Child inherits same configuration and batching behavior")

    print("\n🎯 Key Benefits of Using Batcher:")
    print("-" * 35)
    print("• Automatic batching reduces API calls")
    print("• Token counting prevents API errors")
    print("• Large text splitting with weighted averaging")
    print("• Works seamlessly with both SDK and REST implementations")
    print("• Built-in error handling and retry logic")
    print("• Child batcher creation for hierarchical processing")


async def demonstrate_concept():
    """Demonstrate the concept without making API calls."""
    print("\n🛡️  Batcher Compatibility Demo:")
    print("-" * 30)

    print("   ℹ️  The OpenAIEmbeddingBatcher works with both:")
    print("   • OpenAIEmbeddingsLLMImpl (SDK-based)")
    print("   • OpenAIEmbeddingsRestLLMImpl (REST API-based)")
    print("   ℹ️  Both implementations share the same interface")
    print("   ℹ️  Error handling is built into the LLM implementations")
    print("   ℹ️  Both versions handle API errors gracefully")


if __name__ == "__main__":
    # Check if API key is set
    if os.getenv("OPENAI_API_KEY") is None:
        print("⚠️  Please set your OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("\n   This example will demonstrate the concept without calling OpenAI")

        # Run a basic demo without API calls
        asyncio.run(demonstrate_concept())
    else:
        # Run the full demo
        asyncio.run(main())

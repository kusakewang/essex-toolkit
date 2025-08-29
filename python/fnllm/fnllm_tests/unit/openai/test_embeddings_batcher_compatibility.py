# Copyright (c) 2025 Microsoft Corporation.

"""Tests for the OpenAI embeddings batcher compatibility with REST LLM."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import tiktoken

from fnllm.openai.llm.openai_embeddings_batcher import OpenAIEmbeddingBatcher
from fnllm.openai.llm.openai_embeddings_rest_llm import \
    OpenAIEmbeddingsRestLLMImpl
from fnllm.openai.services.openai_text_service import OpenAITextService
from fnllm.types.metrics import LLMUsageMetrics


class TestEmbeddingsBatcherCompatibility:
    """Test compatibility between batcher and both LLM implementations."""

    @pytest.fixture
    def text_service(self):
        """Create a test text service."""
        encoding = tiktoken.get_encoding("cl100k_base")
        return OpenAITextService(encoding)

    @pytest.fixture
    def rest_llm(self):
        """Create a REST LLM instance."""
        return OpenAIEmbeddingsRestLLMImpl(
            base_url="https://api.openai.com",
            api_key="test-key",
            model="text-embedding-3-small",
        )

    @pytest.fixture
    def mock_response_data(self):
        """Mock response data from OpenAI API."""
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
                }
            ],
            "model": "text-embedding-3-small",
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5
            }
        }

    @pytest.mark.asyncio
    async def test_batcher_with_rest_llm(self, rest_llm, text_service, mock_response_data):
        """Test that the batcher works with REST LLM implementation."""

        # Mock the HTTP client
        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_response.headers = {"content-type": "application/json"}
        mock_response.raise_for_status.return_value = None

        with patch.object(rest_llm._http_client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            # Create batcher with REST LLM
            batcher = OpenAIEmbeddingBatcher(
                llm=rest_llm,
                text_service=text_service,
                max_batch_size=2,
                max_batch_tokens=8192,
            )

            # Test single embedding
            result = await batcher("Test text")

            assert isinstance(result, list)
            assert len(result) == 5  # Length of mock embedding
            assert result == [0.1, 0.2, 0.3, 0.4, 0.5]

            # Verify HTTP call was made
            mock_post.assert_called_once()

    @pytest.mark.asyncio
    async def test_batcher_interface_compatibility(self, rest_llm, text_service):
        """Test that batcher interface works with both LLM types."""

        # Create batcher with REST LLM
        batcher = OpenAIEmbeddingBatcher(
            llm=rest_llm,
            text_service=text_service,
            max_batch_size=2,
            max_batch_tokens=8192,
        )

        # Test that common methods are available
        assert hasattr(batcher, 'child')
        assert hasattr(batcher, 'is_reasoning_model')
        assert callable(batcher)  # Should be callable for batching

        # Test child creation
        child = batcher.child("test-child")
        assert isinstance(child, OpenAIEmbeddingBatcher)

        # Test reasoning model check
        is_reasoning = batcher.is_reasoning_model()
        assert isinstance(is_reasoning, bool)
        assert not is_reasoning  # Embedding models are not reasoning models

    def test_batcher_accepts_both_llm_types(self, text_service):
        """Test that batcher constructor accepts both LLM implementations."""

        # Test with REST LLM
        rest_llm = OpenAIEmbeddingsRestLLMImpl(
            base_url="https://api.openai.com",
            api_key="test-key",
            model="test-model",
        )

        rest_batcher = OpenAIEmbeddingBatcher(
            llm=rest_llm,
            text_service=text_service,
            max_batch_size=2,
            max_batch_tokens=8192,
        )

        assert rest_batcher._llm is rest_llm
        assert rest_batcher.max_batch_size == 2
        assert rest_batcher.max_batch_cost == 8192

        # Both should have the same interface
        assert hasattr(rest_batcher, 'child')
        assert hasattr(rest_batcher, 'is_reasoning_model')
        assert callable(rest_batcher)

    def test_type_annotations(self):
        """Test that type annotations allow both LLM implementations."""
        from fnllm.openai.llm.openai_embeddings_llm import \
            OpenAIEmbeddingsLLMImpl
        from fnllm.openai.types import OpenAIEmbeddingsLLM

        # Both implementations should be assignable to the type alias
        rest_llm = OpenAIEmbeddingsRestLLMImpl(
            base_url="https://api.openai.com",
            api_key="test-key",
            model="test-model",
        )

        # This should not raise type errors in a proper type checker
        llm_var: OpenAIEmbeddingsLLM = rest_llm
        assert llm_var is rest_llm

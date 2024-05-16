"""Test embedding model integration."""

import os

from langchain_together.embeddings import TogetherEmbeddings

os.environ["TOGETHER_API_KEY"] = "foo"


def test_initialization() -> None:
    """Test embedding model initialization."""
    TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")

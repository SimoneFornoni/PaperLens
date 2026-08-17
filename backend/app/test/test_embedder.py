import pytest
from unittest.mock import patch
from pathlib import Path

from backend.app.services.embedder import Embedder

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_vector_size_property():
    SUT = Embedder()
    vector_size = SUT.dense_vector_size

    assert isinstance(vector_size, int)
    assert vector_size > 0

def test_embed_text():
    SUT = Embedder()
    texts = ["This is a test.", "Another test sentence."]
    embeddings = SUT.embed_text(texts)

    assert "dense" in embeddings
    assert "sparse" in embeddings
    assert len(embeddings["dense"]) == len(texts)
    assert len(embeddings["sparse"]) == len(texts)

def test_embed_empty_text():
    SUT = Embedder()
    embeddings = SUT.embed_text([])

    assert embeddings == {"dense": [], "sparse": []}

def test_embed_memory_memory_exeption():
    SUT = Embedder()

    texts = ["This is a test.", "Another test sentence."]
    with patch.object(SUT, "_embed_dense_batch", side_effect=MemoryError("Simulated memory error")):
        with pytest.raises(MemoryError, match="Simulated memory error"):
            SUT.embed_text(texts)

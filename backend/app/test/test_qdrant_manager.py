import pytest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

from backend.app.services.qdrant_manager import QdrantManager

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_upsert_chunks():
    with patch("backend.app.services.qdrant_manager.qc.QdrantClient") as mock_qdrant_client:
        mock_qdrant_client.return_value.collection_exists.return_value = True
        mock_qdrant_client.return_value.upsert.return_value = None
        
        SUT = QdrantManager()
        
        chunks = [
            SimpleNamespace(
                text="Chunk 1",
                context_text="Context 1",
                title="Title 1",
                pdf_hash="hash1",
                section="body",
                page_num=1,
                chunk_idx=0,
                sentence_count=5,
                sentences=[("Sentence 1", "Sentence 2")]
            ),
            SimpleNamespace(
                text="Chunk 2",
                context_text="Context 2",
                title="Title 2",
                pdf_hash="hash2",
                section="body",
                page_num=2,
                chunk_idx=1,
                sentence_count=3,
                sentences=[("Sentence 3", "Sentence 4")]
            )
        ]

        dense_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        sparse_embeddings = [
            {"indices": [1], "values": [0.5]},
            {"indices": [2], "values": [0.6]},
        ]

        result = SUT.upsert_chunks(chunks, dense_embeddings, sparse_embeddings)

    assert result == len(chunks)

def test_upsert_chunks_raises_for_length_mismatch():
    SUT = QdrantManager()
    
    chunks = [SimpleNamespace(text="Chunk 1")]
    dense_embeddings = [[0.1, 0.2]]
    sparse_embeddings = [{"indices": [1], "values": [0.5]}, {"indices": [2], "values": [0.6]}]

    with pytest.raises(ValueError, match="Length of chunks, dense_embeddings, and sparse_embeddings must be the same."):
        SUT.upsert_chunks(chunks, dense_embeddings, sparse_embeddings)
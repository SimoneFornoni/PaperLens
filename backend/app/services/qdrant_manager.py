"""Qdrant handler for vector database operations."""

import logging
import qdrant_client as qc
from qdrant_client.http import models as qm
from typing import List, Dict, Optional
import uuid

from backend.app.utils.sanitizer import sanitize_model_name
from backend.app.models.chunk_models import Chunk

logger = logging.getLogger(__name__)

# TODO: Registry on next commit

class QdrantManager:
    """Manage Qdrant vector database operations for storing and retrieving embeddings."""

    def __init__(self,
                dense_model_name: str = "BAAI/bge-base-en-v1.5", 
                qdrant_url: str = "http://localhost:6333",
                dense_vector_size: int = 768):
        """Initialize the QdrantManager.

        Args:
            dense_model_name (str): Name of the dense model used for embeddings.
            qdrant_url (str): URL of the Qdrant service.
            dense_vector_size (int): Size of the dense vector embeddings.
        """
        self.dense_model_name = dense_model_name
        self.qdrant_url = qdrant_url
        self.qdrant_collection_name = f"PaperLens_{sanitize_model_name(dense_model_name)}"
        self.dense_vector_size = dense_vector_size
        self._client = None # Lazy initialization

    @property
    def client(self) -> qc.QdrantClient:
        """Get the Qdrant client, initializing it if necessary."""
        if self._client is None:
            self._client = qc.QdrantClient(url=self.qdrant_url)
            self._ensure_collections()
        logger.info(f"Qdrant client connection established.")
        return self._client

    def _ensure_collections(self):
        """Ensure that the necessary collections exist in Qdrant."""
        if not self.client.collection_exists(self.qdrant_collection_name):
            self.client.create_collection(
                collection_name=self.qdrant_collection_name,
                hnsw_config=qm.HnswConfigDiff(
                    m=16, 
                    ef_construct=100
                ),
                vectors_config=qm.VectorParams(
                    size=self.dense_vector_size, 
                    distance=qm.Distance.COSINE,
                    data_type=qm.Datatype.FLOAT16,
                    on_disk=True
                ),
                sparse_vectors_config={
                    "sparse_vector": qm.SparseVectorParams(
                        modifier=qm.Modifier.IDF,
                        index=qm.SparseIndexParams(
                            on_disk=True
                        )
                    )
                }
            )

            self.client.create_payload_index(
                collection_name=self.qdrant_collection_name,
                field_name="pdf_hash",
                field_schema=qm.KeywordIndexParams(
                    type="keyword",
                    enable_hnsw=False,  # No need for HNSW index because there are no vector search with filter on pdf_hash
                    on_disk=True
                )
            )

            logger.info(f"Qdrant collection '{self.qdrant_collection_name}' is ready.")
        else:
            logger.info(f"Qdrant collection '{self.qdrant_collection_name}' already exists.")

    def close_client(self):
        """Close the Qdrant client connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("Qdrant client connection closed.")

    def _generate_point_id(self, pdf_hash: str, chunk_idx: Optional[int]) -> str:
        """Generate a unique point ID for a chunk based on its PDF hash and chunk index.
        
        Args:
            pdf_hash (str): The hash of the PDF document.
            chunk_idx (Optional[int]): The index of the chunk within the document.
            
        Returns:
            str: A deterministic UUID string representing the unique point ID for the chunk.
            """
        input_str = f"{pdf_hash}_{chunk_idx}" if chunk_idx is not None else pdf_hash
        NAMESAPCE_PaperLens = uuid.UUID("12345678-1234-5678-1234-567812345678")
        return str(uuid.uuid5(NAMESAPCE_PaperLens, input_str))

    # TODO: check if title is already in use or pdf is already indexed
    def upsert_chunks(self,
                      chunks: List[Chunk],
                      dense_embeddings: List[List[float]],
                      sparse_embeddings: List[Dict[str, List[float]]]) -> int:
        """Upsert chunks into the Qdrant collection.
        
        Args:
            chunks (List[Chunk]): List of Chunk objects to be upserted.
            dense_embeddings (List[List[float]]): List of dense embeddings corresponding to the chunks.
            sparse_embeddings (List[Dict[str, List[float]]]): List of sparse embeddings corresponding to the chunks.
        """

        if not (len(chunks) == len(dense_embeddings) == len(sparse_embeddings)):
            raise ValueError("Length of chunks, dense_embeddings, and sparse_embeddings must be the same.")

        points = []
        for chunk, dense_emb, sparse_emb in zip(chunks, dense_embeddings, sparse_embeddings):
            point_id = self._generate_point_id(chunk.pdf_hash, chunk.chunk_idx)

            vector_config = {
                "dense_vector": dense_emb,
                "sparse_vector": sparse_emb
            }

            point = qm.PointStruct(
                id=point_id,
                vector=vector_config,
                payload={
                    "text": chunk.text,
                    "context_text": chunk.context_text,
                    "title": chunk.title,
                    "pdf_hash": chunk.pdf_hash,
                    "section": chunk.section,
                    "page_num": chunk.page_num,
                    "chunk_idx": chunk.chunk_idx,
                    "sentence_count": chunk.sentence_count,
                    "sentences": chunk.sentences
                }
            )
            points.append(point)

        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch_points = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.qdrant_collection_name,
                points=batch_points
            )

        logger.info(f"Upserted {len(points)} chunks into Qdrant collection '{self.qdrant_collection_name}'.")
        return len(points)
        
        

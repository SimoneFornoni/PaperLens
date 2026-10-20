"""Module for managing vector storage and retrieval in Qdrant."""

import logging
from qdrant_client.http import models as qm
from typing import List, Dict
import uuid

from app.utils.sanitizer import sanitize_model_name
from app.models.domain import Chunk
from app.repositories.qdrant_connection_provider import QdrantConnectionProvider

logger = logging.getLogger(__name__)

class VectorRepository:
    """Manages the vector collection in Qdrant, allowing for chunk storage and retrieval."""

    def __init__(self,
                connection: QdrantConnectionProvider,
                qdrant_url: str = "http://localhost:6333",
                dense_model_name: str = "BAAI/bge-base-en-v1.5", 
                dense_vector_size: int = 768):
        """Initializes the VectorRepository with a Qdrant connection provider and configuration.
        
        Args:
            connection (QdrantConnectionProvider): The connection provider for Qdrant.
            qdrant_url (str): The URL of the Qdrant instance.
            dense_model_name (str): The name of the dense embedding model.
            dense_vector_size (int): The size of the dense vectors.
        """

        self._connection = connection
        self.qdrant_url = qdrant_url
        self.dense_model_name = dense_model_name
        self.dense_vector_size = dense_vector_size
        self.collection_name = f"PaperLens_{sanitize_model_name(dense_model_name)}"
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensures that the vector collection exists in Qdrant, creating it if necessary."""
        if not self._connection.client.collection_exists(self.collection_name):
            self._connection.client.create_collection(
                collection_name=self.collection_name,
                hnsw_config=qm.HnswConfigDiff(
                    m=16, 
                    ef_construct=100
                ),
                vectors_config={
                    "dense_vector": qm.VectorParams(
                        size=self.dense_vector_size, 
                        distance=qm.Distance.COSINE,
                        datatype=qm.Datatype.FLOAT16,
                        on_disk=True
                    )
                },
                sparse_vectors_config={
                    "sparse_vector": qm.SparseVectorParams(
                        modifier=qm.Modifier.IDF,
                        index=qm.SparseIndexParams(
                            on_disk=True
                        )
                    )
                }
            )

            self._connection.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="document_hash",
                field_schema=qm.KeywordIndexParams(
                    type="keyword",
                    enable_hnsw=False,  # No need for HNSW index because there are no vector search with filter on document_hash
                    on_disk=True
                )
            )

            logger.info(f"Qdrant collection '{self.collection_name}' is ready.")

    def _generate_point_id(self, document_hash: str, chunk_idx: int) -> str:
        """Generates a unique point ID for Qdrant.

        Args:
            document_hash (str): The hash of the document.
            chunk_idx (int): The index of the chunk within the document.
            
        Returns:
            str: A deterministic UUID string based on the document hash and chunk index.
        """
        input_str = f"{document_hash}_{chunk_idx}"
        NAMESAPCE_PaperLens = uuid.UUID("12345678-1234-5678-1234-567812345678")
        return str(uuid.uuid5(NAMESAPCE_PaperLens, input_str))

    def upsert_chunks(self,
                      chunks: List[Chunk],
                      dense_embeddings: List[List[float]],
                      sparse_embeddings: List[Dict[str, List[float]]]) -> int:
        """Upserts chunks into the Qdrant collection with their corresponding embeddings.
        
        Args:
            chunks (List[Chunk]): A list of Chunk objects to be upserted.
            dense_embeddings (List[List[float]]): A list of dense embeddings corresponding to the chunks.
            sparse_embeddings (List[Dict[str, List[float]]]): A list of sparse embeddings corresponding to the chunks.
            
        Returns:
            int: The number of chunks successfully upserted.
        """

        if not (len(chunks) == len(dense_embeddings) == len(sparse_embeddings)):
            raise ValueError("Length of chunks, dense_embeddings, and sparse_embeddings must be the same.")

        points = []
        for chunk, dense_emb, sparse_emb in zip(chunks, dense_embeddings, sparse_embeddings):
            point_id = self._generate_point_id(chunk.document_hash, chunk.chunk_idx)

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
                    "document_hash": chunk.document_hash,
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
            self._connection.client.upsert(
                collection_name=self.collection_name,
                points=batch_points
            )

        logger.info(f"Upserted {len(points)} chunks into Qdrant collection '{self.collection_name}'.")
        return len(points)

    def delete_document_chunks(self, document_hash: str) -> None:
        """Delete all vector points belonging to a document in this model's collection.
        
        Args:
            document_hash (str): The hash of the document whose chunks are to be deleted.
        """
        self._connection.client.delete(
            collection_name=self.collection_name,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="document_hash",
                            match=qm.MatchValue(value=document_hash),
                        )
                    ]
                )
            ),
            wait=True,
        )
        
        logger.info(
            "Deleted vector points for document '%s' from collection '%s'.",
            document_hash,
            self.collection_name,
        )

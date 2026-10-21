"""Embedding manager for dense and sparse embeddings."""
import logging
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

from app.utils.device import resolve_device

logger = logging.getLogger(__name__)

DEFAULT_ENCODE_BATCH_SIZE = 32

# The maximum number of tokens to process in a single sequence, 
#  so it matches the reranker model's maximum token limit.
MAX_TOKES = 512

class Embedder:
    """Manages the embedding process for text data, supporting both dense and sparse embeddings."""

    def __init__(self,
                 dense_model_name: str = "BAAI/bge-base-en-v1.5",
                 device: str = None,
                 encode_batch_size: int | None = DEFAULT_ENCODE_BATCH_SIZE):
        """Initialize the Embedder with specified models and parameters.
        
        Args:
            dense_model_name (str): The model name for dense embeddings.
            device (str): The device to run the embedding on (e.g., 'cpu', 'cuda'). Auto-detect CUDA if NONE.
            encode_batch_size (int | None): The batch size for encoding. If None, defaults to DEFAULT_ENCODE_BATCH_SIZE.
        """
        self.dense_model_name = dense_model_name
        self.sparse_model_name = "Qdrant/bm25"
        self.device = resolve_device(device)
        self.encode_batch_size = encode_batch_size or DEFAULT_ENCODE_BATCH_SIZE

        self.dense_model = SentenceTransformer(self.dense_model_name, device=self.device)
        self.dense_model.max_seq_length = MAX_TOKES
        self.sparse_model = SparseTextEmbedding(model_name=self.sparse_model_name, cuda=self.device == "cuda")
        self._dense_vector_size = self._resolve_dense_vector_size()

    @property
    def dense_vector_size(self) -> int:
        """Return the dense embedding dimensionality.
        
        Returns:
            int: The size of the dense embedding vector.
        """
        return self.dense_model.get_embedding_dimension()

    def embed_text(self, all_texts: List[str]) -> Dict[str, List[float]]:
        """Encode a list of texts into dense and sparse embeddings.
        
        Args:
            all_texts (List[str]): A list of text strings to embed.

        Returns:
            Dict[str, List[float]]: A dictionary containing dense and sparse embeddings.
        """
        if not all_texts:
            logger.warning("No texts provided for encoding.")
            return {"dense": [], "sparse": []}

        dense_embeddings = []
        sparse_embeddings = []

        for i in range(0, len(all_texts), self.encode_batch_size):
            batch = all_texts[i:i + self.encode_batch_size]
            try:
                dense_batch = self._embed_dense_batch(batch)
                sparse_batch = self._embed_sparse_batch(batch)
            except (RuntimeError, MemoryError) as exc:
                if not self._is_memory_related_error(exc):
                    raise
    
                fallback_size = max(1, self.encode_batch_size // 2)
                dense_chunks = []
                sparse_batch = []

                for i in range(0, len(batch), fallback_size):
                    fallback_batch = batch[i:i + fallback_size]
                    dense_chunks.append(self._embed_dense_batch(fallback_batch))
                    sparse_batch.extend(self._embed_sparse_batch(fallback_batch))

                dense_batch = np.vstack(dense_chunks)

            dense_embeddings.append(dense_batch)
            sparse_embeddings.extend(sparse_batch)

        if not dense_embeddings:
            return {"dense": [], "sparse": []}

        logger.info(f"Successfully encoded {len(all_texts)} texts into embeddings.")
        return {
            "dense": np.vstack(dense_embeddings).tolist(),
            "sparse": sparse_embeddings
        }

    @staticmethod
    def _is_memory_related_error(exc: Exception) -> bool:
        """Return True when the exception looks like a memory-related model failure.
        
        Args:
            exc (Exception): The exception to check.
            
        Returns:
            bool: True if the exception is memory-related, False otherwise.
        """
        error_str = str(exc).lower()
        return any(
            phrase in error_str
            for phrase in ["out of memory", "buffer size", "cuda", "memory", "onnxruntime"]
        )

    def _embed_dense_batch(self, batch: List[str]) -> np.ndarray:
        """Encode a batch of texts into dense embeddings.
        
        Args:
            batch (List[str]): A list of text strings to embed.
            
        Returns:
            np.ndarray: A 2D array of dense embeddings.
        """
        dense_batch = np.asarray(list(self.dense_model.embed(batch)), dtype=np.float32)
        if dense_batch.ndim == 1:
            dense_batch = np.expand_dims(dense_batch, axis=0)
        return dense_batch

    def _embed_sparse_batch(self, batch: List[str]) -> List[Dict[str, List[float]]]:
        """Encode a batch of texts into sparse embeddings.
        
        Args:
            batch (List[str]): A list of text strings to embed.
            
        Returns:
            List[Dict[str, List[float]]]: A list of sparse embeddings, each represented as a dictionary
                                            with 'indices' and 'values'."""
        return [
            {"indices": res.indices.tolist(), "values": res.values.tolist()}
            for res in self.sparse_model.embed(batch)
        ]
"""Embedding manager for dense and sparse embeddings."""
import logging

from typing import List, Dict
import torch
import numpy as np
from fastembed import TextEmbedding, SparseTextEmbedding

logger = logging.getLogger(__name__)

class Embedder:
    """Manages the embedding process for text data, supporting both dense and sparse embeddings."""

    def __init__(self,
                 dense_model_name: str = "BAAI/bge-base-en-v1.5",
                 sparse_model_name: str = "Qdrant/bm25",
                 device: str = None):
        """Initialize the Embedder with specified models and parameters.
        
        Args:
            dense_model_name (str): The model name for dense embeddings.
            sparse_model_name (str): The model name for sparse embeddings.
            device (str): The device to run the embedding on (e.g., 'cpu', 'cuda'). Auto-detect CUDA if NONE.
        """
        self.dense_model_name = dense_model_name
        self.sparse_model_name = sparse_model_name
        self.device = (device or ("cuda" if torch.cuda.is_available() else "cpu")).lower()

        use_cuda = self.device == "cuda"
        self.dense_model = TextEmbedding(model_name=self.dense_model_name, cuda=use_cuda)
        self.sparse_model = SparseTextEmbedding(model_name=self.sparse_model_name, cuda=use_cuda)
        self._dense_vector_size = self._resolve_dense_vector_size()

    @property
    def dense_vector_size(self) -> int:
        """Return the size of the dense embedding vectors.
        
        Returns:
            int: The dimensionality of the dense embedding vectors.
        """
        return self._dense_vector_size

    def _resolve_dense_vector_size(self) -> int:
        """Determine the dimensionality of the dense embedding vectors.
        
        Returns:
            int: The size of the dense embedding vectors.
        """
        model_info = next(
            info for info in TextEmbedding.list_supported_models()
            if info["model"] == self.dense_model_name
        )
        return model_info["dim"]

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

        batch_size = self._find_safe_batch_size(all_texts)
        dense_embeddings = []
        sparse_embeddings = []

        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i + batch_size]
            try:
                dense_batch = self._embed_dense_batch(batch)
                sparse_batch = self._embed_sparse_batch(batch)
            except (RuntimeError, MemoryError) as exc:
                if not self._is_memory_related_error(exc):
                    raise
    
                fallback_size = max(1, batch_size // 2)
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

    def _find_safe_batch_size(self, 
                              sample_texts: List[str],
                              start_size: int = 2,
                              max_size: int = 128,
                              target_memory_fraction: float = 0.75) -> int:
        """Determine a safe encoding batch size targeting a memory usage fraction.
        
        Args:
            sample_texts (List[str]): A list of sample texts to test encoding.
            start_size (int): The initial batch size to start testing from.
            max_size (int): The maximum batch size to test.
            target_memory_fraction (float): The fraction of available memory to target for usage.
        Returns:
            int: The determined safe batch size for encoding.
        """
        if not sample_texts:
            return start_size

        test_sample = sample_texts[: min(100, len(sample_texts))]

        current_batch_size = start_size
        last_safe_batch_size = start_size

        while current_batch_size <= max_size:
            try:
                test_batch = test_sample[: min(current_batch_size, len(test_sample))]

                if len(test_batch) < current_batch_size and last_safe_batch_size >= len(test_batch):
                    break

                self._embed_dense_batch(test_batch)
                self._embed_sparse_batch(test_batch)
                last_safe_batch_size = current_batch_size
                current_batch_size = min(max_size, int(current_batch_size * 1.5))
            except (RuntimeError, MemoryError, ValueError) as exc:
                if not self._is_memory_related_error(exc):
                    return last_safe_batch_size
                return max(start_size, int(last_safe_batch_size * target_memory_fraction))
            except Exception:
                return last_safe_batch_size

        logger.info(f"Found safe batch size for embedding: {last_safe_batch_size}")
        return max(start_size, int(last_safe_batch_size * target_memory_fraction))

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
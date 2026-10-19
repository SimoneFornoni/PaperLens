"""Qdrant connection provider module."""
import logging
import qdrant_client as qc

logger = logging.getLogger(__name__)

class QdrantConnectionProvider:
    """Provides a connection to the Qdrant service."""

    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        """Initializes the QdrantConnectionProvider with the specified Qdrant URL."""
        self.qdrant_url = qdrant_url
        self._client: qc.QdrantClient | None = None

    @property
    def client(self) -> qc.QdrantClient:
        """Returns the Qdrant client, initializing it if necessary."""
        if self._client is None:
            self._client = qc.QdrantClient(url=self.qdrant_url)
            logger.info(f"Connected to Qdrant at {self.qdrant_url}.")
        return self._client
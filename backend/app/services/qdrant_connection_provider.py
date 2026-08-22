import logging
import qdrant_client as qc

logger = logging.getLogger(__name__)

class QdrantConnectionProvider:

    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        self.qdrant_url = qdrant_url
        self._client: qc.QdrantClient | None = None

    @property
    def client(self) -> qc.QdrantClient:
        if self._client is None:
            self._client = qc.QdrantClient(url=self.qdrant_url)
            logger.info(f"Connected to Qdrant at {self.qdrant_url}.")
        return self._client
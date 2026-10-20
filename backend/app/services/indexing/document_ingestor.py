"""Service to handle the ingestion of documents into the cache."""
import logging
from pathlib import Path
from typing import Dict

from app.services.caches.pdf_cache_handler import PDFCacheHandler
from app.repositories.registry_repository import RegistryRepository
from app.utils.pdf_helpers import compute_hash

logger = logging.getLogger(__name__)

class DocumentIngestor:
    """Class to handle the ingestion of documents into the cache."""

    def __init__(self, 
                 registry_repo: RegistryRepository,
                 output_dir: Path = Path("output"),
                 dense_model_name: str = "BAAI/bge-base-en-v1.5"):
        """Initialize the DocumentIngestor.

        Args:
            registry_repo (RegistryRepository): The registry repository instance.
            output_dir (str): The directory where cached files will be stored.
            dense_model_name (str): The name of the dense model used for indexing.
        """
        self.registry_repo = registry_repo
        self.pdf_cache_handler = PDFCacheHandler(output_dir)
        self.dense_model_name = dense_model_name

    def ingest_document(self, document_path: Path) -> Dict:
        """Ingest a document into the cache and register it in the registry.

        Args:
            document_path (str): The path to the document to be ingested.

        Returns:
            Dict: A dictionary indicating the success of the operation, message, and document hash.
        """
        document_path_obj = Path(document_path)
        if not document_path_obj.exists():
            logger.error(f"File {document_path} does not exist.")
            raise ValueError(f"File {document_path} does not exist.")

        document_hash = compute_hash(document_path_obj)
        document_record = self.registry_repo.is_document_registered(document_hash=document_hash)
        if document_record is not None and self.dense_model_name in document_record.models:
            return {
                "status": "already_indexed",
                "message": f"Document {document_record.title} is already indexed.",
                "document_hash": document_hash,
                "title": document_record.title,
            }
            
        self.pdf_cache_handler.add_pdf(document_path_obj, document_hash=document_hash)
        return {
            "status": "ingested",
            "message": f"Document ingested successfully.",
            "document_hash": document_hash
        }
        
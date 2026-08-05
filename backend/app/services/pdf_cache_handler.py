"""PDF cache handler service for managing PDF files in the cache directory."""

import logging
import os
from pathlib import Path
from typing import List

from ..utils.pdf_helpers import compute_hash

logger = logging.getLogger(__name__)

class PDFCacheHandler:
    """Handles PDFs cache, including saving, retrieving, and deleting PDF files."""

    def __init__(self, cache_dir):
        """Initialize the PDFCacheHandler with a specified cache directory.

        Args:
            cache_dir (str): The directory where PDF files will be cached.
        """

        if not cache_dir:
            raise ValueError("Cache directory must be provided.")

        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

        if not os.path.isdir(self.cache_dir):
            raise ValueError(f"PDF cache directory '{self.cache_dir}' is not a valid directory.")

        logger.info(f"PDF cache directory set to: {self.cache_dir}")

    def ingest_pdf(self, file_path: Path) -> Path:
        """Ingest a PDF file into the cache directory, saving it with a unique hash-based filename."""
        if not file_path.is_file():
            raise ValueError(f"Provided path '{file_path}' is not a valid file.")

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Provided file '{file_path}' is not a PDF.")

        pdf_hash = compute_hash(file_path)
        destination = Path(self.cache_dir) / f"{pdf_hash}.pdf"

        if destination.exists():
            logger.info(f"PDF '{file_path}' already exists in cache as '{destination}'.")
            return destination

        try:
            destination.write_bytes(file_path.read_bytes())
            logger.info(f"PDF '{file_path}' ingested into cache as '{destination}'.")
            return destination
        except Exception as e:
            logger.error(f"Failed to ingest PDF '{file_path}' into cache: {e}")
            raise

    def get_items(self) -> List[str]:
        """Get a list of all PDF files in the cache directory.
        
        Returns:
            List[str]: A list of PDF filenames in the cache directory.
        """
        try:
            return [f.name for f in Path(self.cache_dir).glob("*.pdf")]
        except Exception as e:
            logger.error(f"Failed to list PDF files in cache directory '{self.cache_dir}': {e}")
            raise

    def remove_pdf(self, pdf_hash: str) -> bool:
        """Remove a PDF file from the cache directory based on its hash.

        Args:
            pdf_hash (str): The hash of the PDF file to be removed.
        """
        pdf_path = Path(self.cache_dir) / f"{pdf_hash}.pdf"

        if not pdf_path.exists():
            logger.warning(f"PDF with hash '{pdf_hash}' does not exist in cache.")
            return False

        try:
            pdf_path.unlink()
            logger.info(f"PDF with hash '{pdf_hash}' removed from cache.")
            return True
        except Exception as e:
            logger.error(f"Failed to remove PDF with hash '{pdf_hash}' from cache: {e}")
            raise
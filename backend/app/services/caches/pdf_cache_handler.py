"""PDF cache handler service for managing PDF files in the cache directory."""

from pathlib import Path

from app.services.caches.base_cache_handler import BaseCacheHandler

class PDFCacheHandler(BaseCacheHandler):
    """Handles PDF cache, including saving, retrieving, and deleting PDF files."""

    def __init__(self, output_dir: str | Path):
        super().__init__(output_dir, ".pdf", "PDF")

    def add_pdf(self, file_path: Path, document_hash: str) -> Path:
        """Ingest a PDF file into the cache directory."""
        return self.add_file(file_path, document_hash=document_hash)

    def get_pdf_path(self, pdf_hash: str) -> Path:
        """Retrieve the path of a cached PDF file based on its hash."""
        return self.get_file_path(pdf_hash)

    def remove_pdf(self, pdf_hash: str) -> bool:
        """Remove a PDF file from the cache directory based on its hash."""
        return self.remove_file(pdf_hash)

    def clear_cache(self) -> None:
        """Clear all PDF files from the cache directory."""
        return super().clear_cache()
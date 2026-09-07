"""TEI cache handler service for managing TEI files in the cache directory."""

from pathlib import Path

from app.services.base_cache_handler import BaseCacheHandler

class TEICacheHandler(BaseCacheHandler):
    """Handles TEI cache, including saving, retrieving, and deleting TEI files."""

    def __init__(self, output_dir: str | Path):
        super().__init__(output_dir, ".tei.xml", "TEI")

    def add_tei(self, file_path: Path) -> Path:
        """Ingest a TEI file into the cache directory."""
        return self.add_file(file_path)

    def get_tei_path(self, tei_hash: str) -> Path:
        """Retrieve the path of a cached TEI file based on its hash."""
        return self.get_file_path(tei_hash)

    def remove_tei(self, tei_hash: str) -> bool:
        """Remove a TEI file from the cache directory based on its hash."""
        return self.remove_file(tei_hash)

    def clear_cache(self) -> None:
        """Clear all TEI files from the cache directory."""
        return super().clear_cache()

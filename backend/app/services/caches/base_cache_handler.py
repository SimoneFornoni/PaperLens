"""Generic cache handler for storing and retrieving files by hash."""

import logging
import os
from pathlib import Path
from typing import List
import re

from app.utils.pdf_helpers import compute_hash

logger = logging.getLogger(__name__)

class BaseCacheHandler:
    """Generic cache handler for files of a given extension."""

    def __init__(self, output_dir: str, allowed_suffix: str, cache_type: str):
        """Initialize the cache handler.
        
        Args:
            output_dir (str): The base directory for the cache.
            allowed_suffix (str): The allowed file extension for cached files.
            cache_type (str): A descriptive name for the type of cache (e.g., "PDF", "TEI").
        """
        if not output_dir:
            logger.error("Cache directory must be provided.")
            raise ValueError("Cache directory must be provided.")

        self.cache_dir = os.path.abspath(str(f"{output_dir}/{cache_type.lower()}_cache"))
        os.makedirs(self.cache_dir, exist_ok=True)

        if not os.path.isdir(self.cache_dir):
            logger.error(f"{cache_type} cache directory '{self.cache_dir}' is not a valid directory.")
            raise ValueError(f"{cache_type} cache directory '{self.cache_dir}' is not a valid directory.")

        self.allowed_suffix = allowed_suffix
        self.kind_name = cache_type
        logger.info(f"{cache_type} cache directory set to: {self.cache_dir}")

    def add_file(self, file_path: Path, document_hash: str) -> Path:
        """Adds a file into the cache directory, saving it with a unique hash-based filename.
        
        Args:
            file_path (Path): The path to the file to be cached.
            document_hash (str): The SHA-256 hash of the file content to be used as the cache key.
            
        Returns:
            Path: The path to the cached file in the cache directory.
        """
        if not file_path.is_file():
            logger.error(f"Provided path '{file_path}' is not a valid file.")
            raise ValueError(f"Provided path '{file_path}' is not a valid file.")

        if re.match(r"^[a-fA-F0-9]{64}$", document_hash) is None:
            logger.error(f"Provided hash '{document_hash}' is not a valid SHA-256 hash.")
            raise ValueError(f"Provided hash '{document_hash}' is not a valid SHA-256 hash.")

        current_suffix = "".join(file_path.suffixes).lower()
        if not current_suffix.endswith(self.allowed_suffix.lower()):
            logger.error(f"Provided file '{file_path}' is not a {self.kind_name}. Expected suffix: '{self.allowed_suffix}', got: '{current_suffix}'.")
            raise ValueError(f"Provided file '{file_path}' is not a {self.kind_name}.")

        destination = Path(self.cache_dir) / f"{document_hash}{self.allowed_suffix}"

        if destination.exists():
            logger.warning(f"{self.kind_name} '{file_path}' already exists in cache as '{destination}'.")
            return destination

        try:
            destination.write_bytes(file_path.read_bytes())
            logger.info(f"{self.kind_name} '{file_path}' ingested into cache as '{destination}'.")
            return destination
        except Exception as e:
            logger.error(f"Failed to ingest {self.kind_name.lower()} '{file_path}' into cache: {e}")
            raise

    def get_items(self) -> List[str]:
        """Get a list of cached files.
        
        Returns:
            List[str]: A list of cached file names in the cache directory.
        """
        try:
            return [f.name for f in Path(self.cache_dir).glob(f"*{self.allowed_suffix}")]
        except Exception as e:
            logger.error(f"Failed to list {self.kind_name.lower()} files in cache directory '{self.cache_dir}': {e}")
            raise

    def get_file_path(self, file_hash: str) -> Path:
        """Retrieve the path of a cached file based on its hash.
        
        Args:
            file_hash (str): The hash of the cached file.

        Returns:
            Path: The path to the cached file in the cache directory.
        """
        file_path = Path(self.cache_dir) / f"{file_hash}{self.allowed_suffix}"
        if not file_path.exists():
            logger.error(f"{self.kind_name} with hash '{file_hash}' not found in cache.")
            raise FileNotFoundError(f"{self.kind_name} with hash '{file_hash}' not found in cache.")
        return file_path

    def remove_file(self, file_hash: str) -> bool:
        """Remove a cached file from the cache directory based on its hash.
        
        Args:
            file_hash (str): The hash of the cached file to be removed.
            
        Returns:
            bool: True if the file was successfully removed, False if it did not exist.
        """
        file_path = Path(self.cache_dir) / f"{file_hash}{self.allowed_suffix}"

        if not file_path.exists():
            logger.warning(f"{self.kind_name} with hash '{file_hash}' does not exist in cache.")
            return False

        try:
            file_path.unlink()
            logger.info(f"{self.kind_name} with hash '{file_hash}' removed from cache.")
            return True
        except Exception as e:
            logger.error(f"Failed to remove {self.kind_name.lower()} with hash '{file_hash}' from cache: {e}")
            raise

    def clear_cache(self):
        """Clear all cached files from the cache directory."""
        try:
            for cache_file in Path(self.cache_dir).glob(f"*{self.allowed_suffix}"):
                cache_file.unlink()
            logger.info(f"All {self.kind_name.lower()} files cleared from cache directory '{self.cache_dir}'.")
        except Exception as e:
            logger.error(f"Failed to clear {self.kind_name.lower()} cache directory '{self.cache_dir}': {e}")
            raise

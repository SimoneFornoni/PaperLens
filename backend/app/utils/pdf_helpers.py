from pathlib import Path
import hashlib

def compute_hash(file_path: Path) -> str:
    """Compute the SHA256 hash of a file.

    Args:
        file_path (Path): The path to the file for which to compute the hash.

    Returns:
        str: The computed SHA256 hash of the file.
    """

    if not file_path.is_file():
        raise ValueError(f"Provided path '{file_path}' is not a valid file.")

    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()
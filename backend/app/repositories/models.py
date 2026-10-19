from dataclasses import dataclass, field
from typing import List

@dataclass(frozen=True)
class RegistryEntry:
    """Represents a document with its metadata."""
    title: str
    document_hash: str
    models: frozenset[str] = field(default_factory=frozenset)
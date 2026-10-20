"""Models used in the repositoryies."""

from dataclasses import dataclass, field

@dataclass(frozen=True)
class RegistryEntry:
    """Represents a document with its metadata."""
    title: str
    document_hash: str
    models: frozenset[str] = field(default_factory=frozenset)
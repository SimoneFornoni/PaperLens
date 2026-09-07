"""Data models for representing structured information inside the application."""
from dataclasses import dataclass, field
from typing import List, Tuple

# Models for chunk elaboration

@dataclass
class ProcessedParagraph:
    """Represents a paragraph after processing."""
    text: str
    page_num: int
    sentences: List[Tuple[str, str]] = field(default_factory=list)

@dataclass
class ExtractedChunk:
    """Represents a paragraph extracted from TEI before PDF metadata is attached."""
    text: str
    page_num: int
    chunk_idx: int
    section: str
    sentences: List[Tuple[str, str]] = field(default_factory=list)

@dataclass
class ContextualizedChunk:
    """Represents a chunk together with its generated contextualization."""
    original_text: str
    contextualized_text: str
    context_text: str

@dataclass
class Chunk:
    """Represents a chunk of text with associated metadata."""
    text: str
    context_text: str
    title: str
    document_hash: str
    section: str
    page_num: int
    chunk_idx: int
    sentence_count: int
    sentences: List[Tuple[str, str]] = field(default_factory=list)

# Models for the registry collection in Qdrant

@dataclass(frozen=True)
class DocumentRecord:
    """Represents a document with its metadata."""
    title: str
    document_hash: str
    models: frozenset[str] = field(default_factory=frozenset)
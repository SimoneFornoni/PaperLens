from dataclasses import dataclass, field
from typing import List, Literal

from app.models.domain import Sentence

@dataclass(frozen=True)
class IndexingResult:
    """Outcome of a document indexing attempt."""
    status: Literal["indexed", "already_indexed", "failed"]
    document_hash: str | None = None
    title: str | None = None
    chunk_count: int = 0
    error: str | None = None
    rollback_errors: tuple[str, ...] = ()

    @property
    def succeeded(self) -> bool:
        return self.status != "failed"

@dataclass
class ProcessedParagraph:
    """Represents a paragraph after processing."""
    text: str
    page_num: int
    sentences: List[Sentence] = field(default_factory=list)

@dataclass
class ExtractedChunk:
    """Represents a paragraph extracted from TEI before PDF metadata is attached."""
    text: str
    page_num: int
    chunk_idx: int
    section: str
    sentences: List[Sentence] = field(default_factory=list)

@dataclass
class ContextualizedChunk:
    """Represents a chunk together with its generated contextualization."""
    original_text: str
    contextualized_text: str
    context_text: str
from dataclasses import dataclass, field
from typing import List

from app.models.domain import Sentence

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
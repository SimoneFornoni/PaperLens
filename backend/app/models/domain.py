"""Data models for representing structured information inside the application."""
from dataclasses import dataclass, field
from typing import List

@dataclass
class Sentence:
    """Represents a sentence with its text and coordinates."""
    text: str
    coords: str

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
    sentences: List[Sentence] = field(default_factory=list)
"""Data models for chunk elaboration and extraction."""
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class ProcessedParagraph:
    """Represents a paragraph after processing."""
    text: str
    page_num: int
    sentences: List[Tuple[str, str]] = field(default_factory=list)

@dataclass
class ChunkingState:
    """Carries the mutable state used while building extracted chunks."""
    paragraph_idx: int
    pending_short_para: str
    pending_coords: List[Tuple[int, int]] = field(default_factory=list)

@dataclass
class ExtractedChunk:
    """Represents a paragraph extracted from TEI before PDF metadata is attached."""
    text: str
    page_num: int
    chunk_idx: int
    section: str
    sentences: List[Tuple[str, str]] = field(default_factory=list)

@dataclass
class Chunk:
    """Represents a paragraph-level chunk for QA."""
    text: str
    page_num: int
    chunk_idx: int
    title: str
    pdf_hash: str
    section: str = "body" 
    sentence_count: int = 0 
    sentences: List[Tuple[str, str]] = field(default_factory=list) 
"""Chunker service for processing TEI XML documents into structured text chunks."""
import logging

import xml.etree.ElementTree as ET
from typing import List, Tuple

from ..models.chunk_models import ProcessedParagraph
from ..models.chunk_models import ExtractedChunk

logger = logging.getLogger(__name__)

class Chunker:
    """Processes TEI XML documents to extract structured text chunks."""

    # GROBID TEI namespace
    TEI_NAMESPACE = {"tei": "http://www.tei-c.org/ns/1.0"}

    # Minimum word count before a paragraph is treated as standalone.
    MIN_PARAGRAPH_WORDS = 10

    def extract_text_chunks(self, tei_root: ET.Element) -> Tuple[List[ExtractedChunk], str]:
        """
        Extract text chunks from the TEI XML root element.

        Args:
            tei_root (ET.Element): The root element of the TEI XML document.
            
        Returns:
            Tuple[List[ExtractedParagraph], str]: A tuple containing a list of extracted paragraphs 
                                                  and the full document text.
        """
        if tei_root is None:
            logger.error("TEI root element is None. Cannot extract text chunks.")
            raise ValueError("TEI root element cannot be None.")

        chunks = []
        full_text_parts = []

        paragraph_idx = 0

        # Extract from abstract (each <p> is a paragraph)
        abstract = tei_root.find('.//tei:abstract', self.TEI_NAMESPACE)
        if abstract is not None:
            paragraph_idx = self._collect_paragraphs(
                abstract.findall('.//tei:p', self.TEI_NAMESPACE),
                "abstract",
                chunks,
                full_text_parts,
                paragraph_idx
            )

        # Extract from body
        body = tei_root.find('.//tei:body', self.TEI_NAMESPACE)
        if body is not None:
            for section_div in body.findall('.//tei:div', self.TEI_NAMESPACE):
                head = section_div.find('.//tei:head', self.TEI_NAMESPACE)
                head_text_raw = "".join(head.itertext()).strip() if head is not None else ""
                section_type = self._determine_section_type(head_text_raw.lower())
                
                paragraph_idx = self._collect_paragraphs(
                    section_div.findall('.//tei:p', self.TEI_NAMESPACE),
                    section_type,
                    chunks,
                    full_text_parts,
                    paragraph_idx
                )

        document_text = " ".join(full_text_parts)
        title_text = self._extract_title_from_tei(tei_root)
        if title_text:
            title_block = f"Title: {title_text}"
            document_text = f"{title_block}\n\n{document_text}" if document_text else title_block

        logger.info(f"Extracted {len(chunks)} text chunks from TEI document.")
        return chunks, document_text
        
    def _collect_paragraphs(self,
                        p_elements: List[ET.Element],
                        section_type: str,
                        paragraphs: List[ExtractedChunk],
                        full_text_parts: List[str],
                        paragraph_idx: int) -> int:
        """ Process a list of <p> elements to collect paragraphs, handling short paragraphs.

        Args:
            p_elements (List[ET.Element]): List of <p> elements to process.
            section_type (str): Section type for categorization.
            paragraphs (List[ExtractedParagraph]): List to append extracted paragraphs to.
            full_text_parts (List[str]): List to append full text parts.
            paragraph_idx (int): Index for the next standalone chunk.
        
        Returns:
            int: Index for the next standalone chunk.
        """
        for p_elem in p_elements:
            processed_paragraph = self._process_paragraph(p_elem)
            p_text = processed_paragraph.text
            sentences_coords = processed_paragraph.sentences
            page_num = processed_paragraph.page_num

            if not p_text:
                continue

            if len(p_text.split()) < self.MIN_PARAGRAPH_WORDS and paragraphs:
                previous_chunk = paragraphs[-1]
                previous_chunk.text = f"{previous_chunk.text} {p_text}"
                previous_chunk.sentences.extend(sentences_coords)
                full_text_parts[-1] = f"{full_text_parts[-1]} {p_text}"
                continue

            extracted_para = ExtractedChunk(
                text=p_text,
                page_num=page_num,
                chunk_idx=paragraph_idx,
                section=section_type,
                sentences=sentences_coords
            )
            paragraphs.append(extracted_para)
            full_text_parts.append(p_text)
            paragraph_idx += 1

        return paragraph_idx

    def _process_paragraph(self, p_elem: ET.Element) -> ProcessedParagraph:
        """Process a single <p> element to extract text, sentences, and page number.

        Args:
            p_elem (ET.Element): The <p> element to process.

        Returns:
            ProcessedParagraph: Extracted paragraph text, list of sentences with coordinates
                                and page number.
        """
        senteces_coords = []
        page_num = 0

        for s in p_elem.findall('.//tei:s', self.TEI_NAMESPACE):
            sentence_text = " ".join("".join(s.itertext()).split())
            coords = s.get('coords', '')

            if sentence_text:
                senteces_coords.append((sentence_text, coords))
                if page_num == 0 and coords:
                    try:
                        page_num = int(coords.split(',')[0]) - 1
                    except ValueError: 
                        pass

        # Handle case where no <s> elements are found, but <p> has text
        if not senteces_coords:
            paragraph_text = " ".join("".join(p_elem.itertext()).split())
            if paragraph_text:
                senteces_coords.append((paragraph_text, ""))
                return ProcessedParagraph(text=paragraph_text, page_num=page_num, sentences=senteces_coords)

        paragraph_text = ' '.join([sent for sent, _ in senteces_coords])
        return ProcessedParagraph(text=paragraph_text, page_num=page_num, sentences=senteces_coords)

    def _determine_section_type(self, head_text):
        """Determine section type based on head text, using simple keyword mapping.
        
        Args:
            head_text: Text content of the section head, normalized to lowercase.
        """
        if not head_text:
            return 'body'

        mapping = {
            'abstract': 'abstract', 'introduction': 'introduction',
            'method': 'methods', 'procedure': 'methods',
            'result': 'results', 'discussion': 'discussion',
            'conclusion': 'conclusion'
        }
        for key, value in mapping.items():
            if key in head_text: return value
        return 'body'

    def _extract_title_from_tei(self, tei_root: ET.Element) -> str:
        """Extract document title from TEI header when available.
        
        Args:
            tei_root: Root element of TEI XML.
            
        Returns:
            Extracted title text or empty string if not found.
        """
        title_xpaths = [
            ".//tei:teiHeader/tei:fileDesc/tei:titleStmt/tei:title[@type='main']",
            ".//tei:teiHeader/tei:fileDesc/tei:titleStmt/tei:title",
            ".//tei:teiHeader//tei:titleStmt/tei:title",
            ".//tei:teiHeader//tei:sourceDesc//tei:biblStruct/tei:analytic/tei:title",
            ".//tei:teiHeader//tei:sourceDesc//tei:biblStruct/tei:monogr/tei:title",
            ".//tei:teiHeader//tei:biblStruct/tei:analytic/tei:title",
            ".//tei:teiHeader//tei:biblStruct/tei:monogr/tei:title",
            ".//tei:biblStruct/tei:analytic/tei:title",
            ".//tei:biblStruct/tei:monogr/tei:title",
        ]

        for xpath in title_xpaths:
            for title_elem in tei_root.findall(xpath, self.TEI_NAMESPACE):
                title_text = " ".join("".join(title_elem.itertext()).split())
                if title_text:
                    return title_text

        return ""
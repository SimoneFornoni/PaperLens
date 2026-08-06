import logging

import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict

from ..models.chunk_models import ProcessedParagraph
from ..models.chunk_models import ChunkingState
from ..models.chunk_models import ExtractedChunk


logger = logging.getLogger(__name__)

class Chunker:

    # Reference section patterns to detect bibliography/references
    REFERENCE_PATTERNS = [
        r'^\s*references\s*$',
        r'^\s*bibliography\s*$',
        r'^\s*works\s+cited\s*$',
        r'^\s*literature\s+cited\s*$',
    ]
    
    # Section types to include in chunking (can be customized)
    CONTENT_SECTIONS = {
        'body': True,
        'abstract': True,
        'introduction': True,
        'conclusion': True,
        'results': True,
        'methods': True,
        'discussion': True,
    }

    # Minimum word count before a paragraph is treated as standalone.
    MIN_PARAGRAPH_WORDS = 10

    def __init__(self):
        """Initialize the Chunker."""
        logger.info("Initialized Chunker for TEI XML processing.")

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

        # Define TEI namespace
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

        # Buffer to handle short paragraphs and paragraph index
        chunking_state = ChunkingState(
            paragraph_idx=0,
            pending_short_para="",
            pending_coords=[]
        )

        # Extract from abstract (each <p> is a paragraph)
        abstract = tei_root.find('.//tei:abstract', ns)
        if abstract is not None:
            chunking_state = self._collect_paragraphs(
                abstract.findall('.//tei:p', ns),
                "abstract",
                chunks,
                full_text_parts,
                chunking_state,
                ns,
            )

        body = tei_root.find('.//tei:body', ns)
        if body is not None:
            for section_div in body.findall('.//tei:div', ns):
                # Determine section type based on head element
                head = section_div.find('.//tei:head', ns)
                head_text_raw = "".join(head.itertext()).strip() if head is not None else ""
                section_type = self._determine_section_type(head_text_raw.lower())
                
                chunking_state = self._collect_paragraphs(
                    section_div.findall('.//tei:p', ns),
                    section_type,
                    chunks,
                    full_text_parts,
                    chunking_state,
                    ns,
                )

        if chunking_state.pending_short_para and chunks:
            last_para = chunks[-1]
            last_para.text = f"{last_para.text} {chunking_state.pending_short_para}"
            last_para.sentences.extend(chunking_state.pending_coords)
            full_text_parts[-1] = f"{full_text_parts[-1]} {chunking_state.pending_short_para}"

        document_text = " ".join(full_text_parts)
        title_text = self._extract_title_from_tei(tei_root, ns)
        if title_text:
            title_block = f"Title: {title_text}"
            document_text = f"{title_block}\n\n{document_text}" if document_text else title_block
        
        return chunks, document_text
        
    def _collect_paragraphs(self,
                        p_elements: List[ET.Element],
                        section_type: str,
                        paragraphs: List[ExtractedChunk],
                        full_text_parts: List[str],
                        chunking_state: ChunkingState,
                        ns: Dict[str, str]) -> ChunkingState:
        """ Process a list of <p> elements to collect paragraphs, handling short paragraphs.

        Args:
            p_elements (List[ET.Element]): List of <p> elements to process.
            section_type (str): Section type for categorization.
            paragraphs (List[ExtractedParagraph]): List to append extracted paragraphs to.
            full_text_parts (List[str]): List to append full text parts.
            paragraph_idx (int): Current paragraph index.
            pending_short_para (str): Buffer for short paragraph text that may need to be merged.
            pending_coords (List[Tuple[int, int]]): Buffer for coordinates of sentences in the pending short paragraph.
            ns (Dict[str, str]): Namespace dictionary for XML parsing.
        
        Returns:
            Tuple[CollectedParagraph]: Updated paragraph index, pending short paragraph text
                                       and pending coordinates.
        """
        for p_elem in p_elements:
            processed_paragraph = self._process_paragraph(p_elem, ns)
            p_text = processed_paragraph.text
            sentences_coords = processed_paragraph.sentences
            page_num = processed_paragraph.page_num

            if not p_text:
                continue

            extracted_para = ExtractedChunk(
                text=p_text,
                page_num=page_num,
                chunk_idx=chunking_state.paragraph_idx,
                section=section_type,
                sentences=sentences_coords
            )
            paragraphs.append(extracted_para)
            full_text_parts.append(p_text)
            chunking_state.paragraph_idx += 1

        return chunking_state

    def _process_paragraph(self, p_elem: ET.Element, ns: Dict[str, str]) -> ProcessedParagraph:
        """Process a single <p> element to extract text, sentences, and page number.

        Args:
            p_elem (ET.Element): The <p> element to process.
            ns (Dict[str, str]): Namespace dictionary for XML parsing.

        Returns:
            ProcessedParagraph: Extracted paragraph text, list of sentences with coordinates
                                and page number.
        """
        senteces_coords = []
        page_num = 0

        for s in p_elem.findall('.//tei:s', ns):
            sentence_text = " ".join("".join(s.itertext()).split())
            coords = s.get('coords', '')

            if sentence_text:
                senteces_coords.append((sentence_text, coords))
                if page_num == 0 and coords:
                    try:
                        page_num = int(coords.split(',')[0]) - 1
                    except ValueError: 
                        pass

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

    def _extract_title_from_tei(self, tei_root: ET.Element, ns: Dict[str, str]) -> str:
        """Extract document title from TEI header when available.
        
        Args:
            tei_root: Root element of TEI XML.
            ns: Namespace dictionary for XML parsing.
            
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
            for title_elem in tei_root.findall(xpath, ns):
                title_text = " ".join("".join(title_elem.itertext()).split())
                if title_text:
                    return title_text

        return ""
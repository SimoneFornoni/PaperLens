"""PDF parsing and text extraction service using GROBID."""

import logging
import threading
import requests
import shutil
import tempfile
from pathlib import Path

import xml.etree.ElementTree as ET
from grobid_client.grobid_client import GrobidClient

from .tei_cache_handler import TEICacheHandler

logger = logging.getLogger(__name__)

class GROBIDManager:
    """Handles PDF parsing and text extraction using GROBID."""
    
    # Serialize calls to GROBID to avoid exhausting its internal pool
    GROBID_LOCK = threading.Lock()

    def __init__(self, 
                grobid_url: str = "http://localhost:8070",
                grobid_timeout: int = 180,
                output_dir: str = "output"):
        """Initialize the PDFProcessor.

        Args:
            grobid_url (str): The URL of the GROBID service.
            grobid_timeout (int): Timeout for GROBID requests in seconds.
        """
        self.grobid_url = grobid_url
        self.grobid_timeout = grobid_timeout
        self.grobid_client = None  # Lazy initialization

        self.tei_cache_handler = TEICacheHandler(output_dir)

        logger.info(f"Initialized PDFProcessor with GROBID URL: {self.grobid_url}")

    def _is_alive(self) -> bool:
        """Check if the GROBID service is alive.

        Returns:
            bool: True if GROBID is alive, False otherwise.
        """
        try:
            # Implement a simple health check for GROBID
            response = requests.get(f"{self.grobid_url}/api/isalive", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"GROBID health check failed: {e}")
            return False

    def parse_pdf(self, pdf_path: Path) -> ET.Element:
        """Parse a single PDF fileusing GROBID and return TEI XML root.
        
        Args:
            pdf_path (str): Path to the PDF file.
            
        Returns:
            ET.Element: XML Element root of the TEI document.
        """
        if pdf_path is None or pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Provided path '{pdf_path}' is not a valid file.")
        
        try:
            if not self._is_alive():
                message = f"GROBID service is not reachable at {self.grobid_url}. Start GROBID and try again."
                raise ConnectionError(message)

            if self.grobid_client is None:
                self.grobid_client = GrobidClient(grobid_server=self.grobid_url)

            with self.GROBID_LOCK:
                try:
                    in_dir_path = Path(tempfile.mkdtemp(prefix="grobid_in_"))
                    out_dir_path = Path(tempfile.mkdtemp(prefix="grobid_out_"))

                    base = pdf_path.name
                    temp_pdf = in_dir_path / base
                    shutil.copy2(pdf_path, temp_pdf)

                    # Process with sentence segmentation and coordinates
                    self.grobid_client.process(
                        service="processFulltextDocument",
                        input_path=str(in_dir_path),
                        output=str(out_dir_path),
                        n=1,
                        tei_coordinates=True,
                        segment_sentences=True
                    )

                    expected = f"{pdf_path.stem}.tei.xml"
                    tei_path = out_dir_path / expected
                    if not tei_path.exists():
                        candidates = [
                            candidate
                            for candidate in out_dir_path.iterdir()
                            if candidate.name.endswith(".tei.xml")
                        ]
                        tei_path = candidates[0] if candidates else None

                    if tei_path and tei_path.exists():
                        content = tei_path.read_bytes()
                        root = ET.fromstring(content)
                        try:
                            _ = self.tei_cache_handler.ingest_tei(tei_path)
                        except Exception as cache_error:
                            logger.warning(f"TEI cache write failed for {pdf_path}: {cache_error}")

                        return root
                    else:
                        logger.warning(f"GROBID client did not produce TEI for {pdf_path}")
                        return None
                except Exception as e:
                    logger.error(f"Error during GROBID processing for {pdf_path}: {e}")
                    raise
                finally:
                    shutil.rmtree(in_dir_path, ignore_errors=True)
                    shutil.rmtree(out_dir_path, ignore_errors=True)
                
        except ConnectionError as conn_err:
            logger.error(f"GROBID connection error: {conn_err}")
            raise
        except Exception as e:
            logger.error(f"Failed to parse PDF '{pdf_path}': {e}")
            raise
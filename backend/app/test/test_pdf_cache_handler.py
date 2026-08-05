from pathlib import Path
import sys
from unittest.mock import Mock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.pdf_cache_handler import PDFCacheHandler


def test_ingest_pdf_with_mocks(tmp_path):
    cache_dir = tmp_path / "pdf_cache"
    SUT = PDFCacheHandler(cache_dir)

    mocked_file = Mock(spec=Path)
    mocked_file.is_file.return_value = True
    mocked_file.suffix = ".pdf"
    mocked_file.read_bytes.return_value = b"%PDF-1.4\n%EOF"

    with patch("backend.app.services.pdf_cache_handler.compute_hash", return_value="fakehash"):
        cached_pdf_path = SUT.ingest_pdf(mocked_file)

    assert cached_pdf_path.name in SUT.get_items()


def test_ingest_pdf_raises_for_missing_file(tmp_path):
    cache_dir = tmp_path / "pdf_cache"
    SUT = PDFCacheHandler(cache_dir)

    missing_file = Mock(spec=Path)
    missing_file.is_file.return_value = False
    missing_file.suffix = ".pdf"

    with pytest.raises(ValueError, match="not a valid file"):
        SUT.ingest_pdf(missing_file)


def test_ingest_pdf_raises_for_non_pdf_file(tmp_path):
    cache_dir = tmp_path / "pdf_cache"
    handler = PDFCacheHandler(cache_dir)

    non_pdf_file = Mock(spec=Path)
    non_pdf_file.is_file.return_value = True
    non_pdf_file.suffix = ".txt"

    with pytest.raises(ValueError, match="not a PDF"):
        handler.ingest_pdf(non_pdf_file)
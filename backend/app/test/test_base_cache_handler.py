import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from backend.app.services.pdf_cache_handler import PDFCacheHandler
from backend.app.services.tei_cache_handler import TEICacheHandler

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_ingest_pdf(tmp_path):
    cache_dir = tmp_path
    SUT = PDFCacheHandler(cache_dir)

    mocked_file = Mock(spec=Path)
    mocked_file.is_file.return_value = True
    mocked_file.suffixes = [".pdf"]
    mocked_file.read_bytes.return_value = b"%PDF-1.4\n%EOF"

    with patch("backend.app.services.base_cache_handler.compute_hash", return_value="fakehash"):
        cached_pdf_path = SUT.ingest_pdf(mocked_file)

    assert cached_pdf_path.name in SUT.get_items()

def test_ingest_tei(tmp_path):
    cache_dir = tmp_path
    SUT = TEICacheHandler(cache_dir)

    mocked_file = Mock(spec=Path)
    mocked_file.is_file.return_value = True
    mocked_file.suffixes = [".tei", ".xml"]
    mocked_file.read_bytes.return_value = b"<TEI>...</TEI>"

    with patch("backend.app.services.base_cache_handler.compute_hash", return_value="fakehash"):
        cached_tei_path = SUT.ingest_tei(mocked_file)

    assert cached_tei_path.name in SUT.get_items()

def test_ingest_file_raises_for_missing_file(tmp_path):
    cache_dir = tmp_path
    SUT = PDFCacheHandler(cache_dir)

    missing_file = Mock(spec=Path)
    missing_file.is_file.return_value = False
    missing_file.suffixes = [".pdf"]

    with pytest.raises(ValueError, match="not a valid file"):
        SUT.ingest_pdf(missing_file)


def test_ingest_pdf_raises_for_non_pdf_file(tmp_path):
    cache_dir = tmp_path
    handler = PDFCacheHandler(cache_dir)

    non_pdf_file = Mock(spec=Path)
    non_pdf_file.is_file.return_value = True
    non_pdf_file.suffixes = [".txt"]

    with pytest.raises(ValueError, match="not a PDF"):
        handler.ingest_pdf(non_pdf_file)

def test_get_items_returns_correct_list(tmp_path):
    cache_dir = tmp_path
    SUT = PDFCacheHandler(cache_dir)

    mocked_file1 = Mock(spec=Path)
    mocked_file1.is_file.return_value = True
    mocked_file1.suffixes = [".pdf"]
    mocked_file1.read_bytes.return_value = b"%PDF-1.4\n%EOF"

    mocked_file2 = Mock(spec=Path)
    mocked_file2.is_file.return_value = True
    mocked_file2.suffixes = [".pdf"]
    mocked_file2.read_bytes.return_value = b"%PDF-1.4\n%EOF"

    with patch("backend.app.services.base_cache_handler.compute_hash", side_effect=["hash1", "hash2"]):
        SUT.ingest_pdf(mocked_file1)
        SUT.ingest_pdf(mocked_file2)

    items = SUT.get_items()
    assert len(items) == 2
    assert any("hash1.pdf" in item for item in items)
    assert any("hash2.pdf" in item for item in items)

def test_get_file_path(tmp_path):
    cache_dir = tmp_path
    SUT = PDFCacheHandler(cache_dir)

    mocked_file = Mock(spec=Path)
    mocked_file.is_file.return_value = True
    mocked_file.suffixes = [".pdf"]
    mocked_file.read_bytes.return_value = b"%PDF-1.4\n%EOF"

    with patch("backend.app.services.base_cache_handler.compute_hash", return_value="fakehash"):
        cached_pdf_path = SUT.ingest_pdf(mocked_file)

    retrieved_path = SUT.get_pdf_path("fakehash")
    assert retrieved_path == cached_pdf_path

def test_remove_file(tmp_path):
    cache_dir = tmp_path
    SUT = PDFCacheHandler(cache_dir)

    mocked_file = Mock(spec=Path)
    mocked_file.is_file.return_value = True
    mocked_file.suffixes = [".pdf"]
    mocked_file.read_bytes.return_value = b"%PDF-1.4\n%EOF"

    with patch("backend.app.services.base_cache_handler.compute_hash", return_value="fakehash"):
        cached_pdf_path = SUT.ingest_pdf(mocked_file)

    assert cached_pdf_path.name in SUT.get_items()

    removed = SUT.remove_pdf("fakehash")
    assert removed is True
    assert cached_pdf_path.name not in SUT.get_items()

def test_remove_nonexistent_file(tmp_path):
    cache_dir = tmp_path
    SUT = PDFCacheHandler(cache_dir)

    removed = SUT.remove_pdf("nonexistenthash")
    assert removed is False

def test_clear_cache(tmp_path):
    cache_dir = tmp_path
    SUT = PDFCacheHandler(cache_dir)

    mocked_file1 = Mock(spec=Path)
    mocked_file1.is_file.return_value = True
    mocked_file1.suffixes = [".pdf"]
    mocked_file1.read_bytes.return_value = b"%PDF-1.4\n%EOF"

    mocked_file2 = Mock(spec=Path)
    mocked_file2.is_file.return_value = True
    mocked_file2.suffixes = [".pdf"]
    mocked_file2.read_bytes.return_value = b"%PDF-1.4\n%EOF"

    with patch("backend.app.services.base_cache_handler.compute_hash", side_effect=["hash1", "hash2"]):
        SUT.ingest_pdf(mocked_file1)
        SUT.ingest_pdf(mocked_file2)

    assert len(SUT.get_items()) == 2

    SUT.clear_cache()

    assert len(SUT.get_items()) == 0
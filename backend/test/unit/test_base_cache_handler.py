import pytest
from unittest.mock import Mock
from pathlib import Path

from app.services.caches.pdf_cache_handler import PDFCacheHandler
from app.services.caches.tei_cache_handler import TEICacheHandler

# import sys
# PROJECT_ROOT = Path(__file__).resolve().parents[3]
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

HASH_1 = "1" * 64
HASH_2 = "2" * 64
FAKE_HASH = "a" * 64
NONEXISTENT_HASH = "b" * 64

def test_ingest_pdf(tmp_path):
    cache_dir = tmp_path
    SUT = PDFCacheHandler(cache_dir)

    mocked_file = Mock(spec=Path)
    mocked_file.is_file.return_value = True
    mocked_file.suffixes = [".pdf"]
    mocked_file.read_bytes.return_value = b"%PDF-1.4\n%EOF"

    cached_pdf_path = SUT.add_pdf(mocked_file, document_hash=FAKE_HASH)

    assert cached_pdf_path.name in SUT.get_items()

def test_ingest_tei(tmp_path):
    cache_dir = tmp_path
    SUT = TEICacheHandler(cache_dir)

    mocked_file = Mock(spec=Path)
    mocked_file.is_file.return_value = True
    mocked_file.suffixes = [".tei", ".xml"]
    mocked_file.read_bytes.return_value = b"<TEI>...</TEI>"

    cached_tei_path = SUT.add_tei(mocked_file, document_hash=FAKE_HASH)

    assert cached_tei_path.name in SUT.get_items()
    assert cached_tei_path.name == f"{FAKE_HASH}.tei.xml"

def test_ingest_grobid_tei(tmp_path):
    cache_dir = tmp_path
    SUT = TEICacheHandler(cache_dir)

    mocked_file = Mock(spec=Path)
    mocked_file.is_file.return_value = True
    mocked_file.suffixes = [".grobid", ".tei", ".xml"]
    mocked_file.read_bytes.return_value = b"<TEI>...</TEI>"

    cached_tei_path = SUT.add_tei(mocked_file, document_hash=FAKE_HASH)

    assert cached_tei_path.name in SUT.get_items()

def test_ingest_file_raises_for_missing_file(tmp_path):
    cache_dir = tmp_path
    SUT = PDFCacheHandler(cache_dir)

    missing_file = Mock(spec=Path)
    missing_file.is_file.return_value = False
    missing_file.suffixes = [".pdf"]

    with pytest.raises(ValueError, match="not a valid file"):
        SUT.add_pdf(missing_file, document_hash=FAKE_HASH)


def test_ingest_pdf_raises_for_non_pdf_file(tmp_path):
    cache_dir = tmp_path
    handler = PDFCacheHandler(cache_dir)

    non_pdf_file = Mock(spec=Path)
    non_pdf_file.is_file.return_value = True
    non_pdf_file.suffixes = [".txt"]

    with pytest.raises(ValueError, match="not a PDF"):
        handler.add_pdf(non_pdf_file, document_hash=FAKE_HASH)

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

    SUT.add_pdf(mocked_file1, document_hash=HASH_1)
    SUT.add_pdf(mocked_file2, document_hash=HASH_2)

    items = SUT.get_items()
    assert len(items) == 2
    assert any(f"{HASH_1}.pdf" in item for item in items)
    assert any(f"{HASH_2}.pdf" in item for item in items)

def test_get_file_path(tmp_path):
    cache_dir = tmp_path
    SUT = PDFCacheHandler(cache_dir)

    mocked_file = Mock(spec=Path)
    mocked_file.is_file.return_value = True
    mocked_file.suffixes = [".pdf"]
    mocked_file.read_bytes.return_value = b"%PDF-1.4\n%EOF"

    cached_pdf_path = SUT.add_pdf(mocked_file, document_hash=FAKE_HASH)

    retrieved_path = SUT.get_pdf_path(FAKE_HASH)
    assert retrieved_path == cached_pdf_path

def test_remove_file(tmp_path):
    cache_dir = tmp_path
    SUT = PDFCacheHandler(cache_dir)

    mocked_file = Mock(spec=Path)
    mocked_file.is_file.return_value = True
    mocked_file.suffixes = [".pdf"]
    mocked_file.read_bytes.return_value = b"%PDF-1.4\n%EOF"

    cached_pdf_path = SUT.add_pdf(mocked_file, document_hash=FAKE_HASH)

    assert cached_pdf_path.name in SUT.get_items()

    removed = SUT.remove_pdf(FAKE_HASH)
    assert removed is True
    assert cached_pdf_path.name not in SUT.get_items()

def test_remove_nonexistent_file(tmp_path):
    cache_dir = tmp_path
    SUT = PDFCacheHandler(cache_dir)

    removed = SUT.remove_pdf(NONEXISTENT_HASH)
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

    SUT.add_pdf(mocked_file1, document_hash=HASH_1)
    SUT.add_pdf(mocked_file2, document_hash=HASH_2)

    assert len(SUT.get_items()) == 2

    SUT.clear_cache()

    assert len(SUT.get_items()) == 0
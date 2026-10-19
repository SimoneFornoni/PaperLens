"""Test cases for the DocumentIngestor service."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.indexing.document_ingestor import DocumentIngestor
from app.models.domain import DocumentRecord


def test_ingest_document(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%EOF")

    mock_registry_repo = Mock()
    mock_registry_repo.is_document_registered.return_value = None  # Document not yet registered

    mock_pdf_cache_handler = Mock()
    mock_pdf_cache_handler.add_pdf = Mock()

    SUT = DocumentIngestor(registry_repo=mock_registry_repo, output_dir=str(tmp_path))
    SUT.pdf_cache_handler = mock_pdf_cache_handler

    with patch("app.services.document_ingestor.compute_hash", return_value="hash1"):
        result = SUT.ingest_document(str(pdf_path))

    assert "ingested successfully" in result["message"]
    assert result["document_hash"] == "hash1"
    mock_pdf_cache_handler.add_pdf.assert_called_once()
    mock_registry_repo.is_document_registered.assert_called_once_with(document_hash="hash1")


def test_ingest_document_file_not_found():
    mock_registry_repo = Mock()

    SUT = DocumentIngestor(registry_repo=mock_registry_repo, output_dir="output")

    with patch("app.services.document_ingestor.compute_hash", return_value="hash1"):
        with pytest.raises(ValueError, match="File non_existent.pdf does not exist."):
            SUT.ingest_document("non_existent.pdf")

def test_ingest_document_already_registered(tmp_path):
    """Test ingestion of a document that is already registered."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%EOF")

    mock_doc_record = DocumentRecord(
        title="already_indexed_paper",
        document_hash="hash1",
        models=frozenset(["BAAI/bge-base-en-v1.5"])
    )

    mock_registry_repo = Mock()
    mock_registry_repo.is_document_registered.return_value = mock_doc_record

    mock_pdf_cache_handler = Mock()

    SUT = DocumentIngestor(
        registry_repo=mock_registry_repo,
        output_dir=str(tmp_path),
        dense_model_name="BAAI/bge-base-en-v1.5"
    )
    SUT.pdf_cache_handler = mock_pdf_cache_handler

    with patch("app.services.document_ingestor.compute_hash", return_value="hash1"):
        result = SUT.ingest_document(str(pdf_path))

    assert "already indexed" in result["message"]
    assert "already_indexed_paper" in result["message"]
    mock_pdf_cache_handler.add_pdf.assert_not_called()

def test_ingest_document_with_different_model(tmp_path):
    """Test ingestion of a document registered with a different model."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%EOF")

    mock_doc_record = DocumentRecord(
        title="paper_with_different_model",
        document_hash="hash1",
        models=frozenset(["some-other-model"])
    )

    mock_registry_repo = Mock()
    mock_registry_repo.is_document_registered.return_value = mock_doc_record

    mock_pdf_cache_handler = Mock()
    mock_pdf_cache_handler.add_pdf = Mock()

    SUT = DocumentIngestor(
        registry_repo=mock_registry_repo,
        output_dir=str(tmp_path),
        dense_model_name="BAAI/bge-base-en-v1.5"
    )
    SUT.pdf_cache_handler = mock_pdf_cache_handler

    with patch("app.services.document_ingestor.compute_hash", return_value="hash1"):
        result = SUT.ingest_document(str(pdf_path))

    assert "ingested successfully" in result["message"]
    assert result["document_hash"] == "hash1"
    mock_registry_repo.is_document_registered.assert_called_once_with(document_hash="hash1")
    mock_pdf_cache_handler.add_pdf.assert_called_once()

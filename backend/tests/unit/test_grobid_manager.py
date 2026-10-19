import pytest
from unittest.mock import patch
from pathlib import Path

from app.services.indexing.pdf_parser import GROBIDManager

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_parse_pdf(tmp_path):
    SUT = GROBIDManager(grobid_url="http://localhost:8070", output_dir=tmp_path)

    mocked_pdf = tmp_path / "test.pdf"
    mocked_pdf.write_bytes(b"%PDF-1.4\n%EOF")

    input_dir = tmp_path / "grobid_in"
    output_dir = tmp_path / "grobid_out"
    input_dir.mkdir()
    output_dir.mkdir()

    def write_tei_result(*, output, **kwargs):
        tei_path = Path(output) / "test.tei.xml"
        tei_path.write_text("<TEI><text>ok</text></TEI>", encoding="utf-8")
        return None

    with patch.object(SUT, '_is_alive', return_value=True), \
         patch('app.services.grobid_manager.compute_hash', return_value='pdfhash'), \
         patch.object(SUT, 'tei_cache_handler') as mock_cache_handler, \
         patch('app.services.grobid_manager.tempfile.mkdtemp', side_effect=[str(input_dir), str(output_dir)]), \
         patch('app.services.grobid_manager.GrobidClient') as mock_grobid_client:
        mock_grobid_instance = mock_grobid_client.return_value
        mock_grobid_instance.process.side_effect = write_tei_result

        _ = SUT.parse_pdf(mocked_pdf)

    assert mock_grobid_instance.process.called
    mock_cache_handler.add_tei.assert_called_once_with(
        Path(output_dir) / "test.tei.xml", document_hash="pdfhash"
    )

def test_parse_pdf_raises_for_non_pdf_file(tmp_path):
    SUT = GROBIDManager(grobid_url="http://localhost:8070", output_dir=tmp_path)

    non_pdf_file = tmp_path / "test.txt"
    non_pdf_file.write_text("This is not a PDF file.", encoding="utf-8")

    with pytest.raises(ValueError, match="not a valid file"):
           SUT.parse_pdf(non_pdf_file)

def test_parse_pdf_raises_for_unreachable_grobid(tmp_path):
    SUT = GROBIDManager(grobid_url="http://localhost:8070", output_dir=tmp_path)

    mocked_pdf = tmp_path / "test.pdf"
    mocked_pdf.write_bytes(b"%PDF-1.4\n%EOF")

    with patch.object(SUT, '_is_alive', return_value=False):
        with pytest.raises(ConnectionError, match="GROBID service is not reachable"):
                SUT.parse_pdf(mocked_pdf)

def test_parse_pdf_raises_for_grobid_client_exception(tmp_path):
    SUT = GROBIDManager(grobid_url="http://localhost:8070", output_dir=tmp_path)

    mocked_pdf = tmp_path / "test.pdf"
    mocked_pdf.write_bytes(b"%PDF-1.4\n%EOF")

    input_dir = tmp_path / "grobid_in"
    output_dir = tmp_path / "grobid_out"
    input_dir.mkdir()
    output_dir.mkdir()

    with patch.object(SUT, '_is_alive', return_value=True), \
         patch('app.services.grobid_manager.tempfile.mkdtemp', side_effect=[str(input_dir), str(output_dir)]), \
         patch('app.services.grobid_manager.GrobidClient') as mock_grobid_client:
        mock_grobid_instance = mock_grobid_client.return_value
        mock_grobid_instance.process.side_effect = Exception("GROBID processing error")

        with pytest.raises(Exception, match="GROBID processing error"):
                        SUT.parse_pdf(mocked_pdf)
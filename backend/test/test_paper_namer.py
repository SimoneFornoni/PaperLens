import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import Mock

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.paper_namer import PaperNamer
from app.models.domain import DocumentRecord

def test_resolve_standard_paper_name(tmp_path):
    mock_registry_repo = Mock()
    mock_registry_repo.is_document_registered.return_value = None  # Document not yet registered
    mock_registry_repo.is_title_registered.return_value = False
    mock_registry_repo.reserve_title = Mock()
    SUT = PaperNamer(mock_registry_repo)

    tei_xml_content = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
        <teiHeader>
            <fileDesc>
                <titleStmt>
                    <title>Sample Paper Title</title>
                    <author>John Doe</author>
                </titleStmt>
                <publicationStmt>
                    <date when="2024">2024</date>
                </publicationStmt>
            </fileDesc>
        </teiHeader>
    </TEI>
    """

    tei_xml_path = tmp_path / "sample.tei.xml"
    tei_xml_path.write_text(tei_xml_content, encoding="utf-8")
    standard_name = SUT.resolve_standard_paper_name(ET.fromstring(tei_xml_content), document_hash="hash1")

    assert standard_name == "doe2024sample"
    mock_registry_repo.reserve_title.assert_called_once_with("doe2024sample")


def test_resolve_standard_paper_name_duplicate(tmp_path):
    mock_registry_repo_instance = Mock()
    mock_registry_repo_instance.is_document_registered.return_value = None  # Document not yet registered
    mock_registry_repo_instance.is_title_registered.side_effect = [False, True, False]
    mock_registry_repo_instance.reserve_title = Mock()

    SUT = PaperNamer(mock_registry_repo_instance)

    tei_xml_content = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
        <teiHeader>
            <fileDesc>
                <titleStmt>
                    <title>Sample Paper Title</title>
                    <author>John Doe</author>
                </titleStmt>
                <publicationStmt>
                    <date when="2024">2024</date>
                </publicationStmt>
            </fileDesc>
        </teiHeader>
    </TEI>
    """

    tei_xml_path = tmp_path / "sample.tei.xml"
    tei_xml_path.write_text(tei_xml_content, encoding="utf-8")
    tei_element = ET.fromstring(tei_xml_content)
    standard_name_1 = SUT.resolve_standard_paper_name(tei_element, document_hash="hash1")
    standard_name_2 = SUT.resolve_standard_paper_name(tei_element, document_hash="hash2")

    assert standard_name_1 == "doe2024sample"
    assert standard_name_2 == "doe2024sample_1"
    assert mock_registry_repo_instance.reserve_title.call_count == 2

def test_resolve_paper_name_for_registered_document(tmp_path):
    """Test that when a document is already registered, its title is returned."""
    mock_registry_repo = Mock()
    mock_doc_record = DocumentRecord(
        title="existing_paper",
        document_hash="hash_existing",
        models=frozenset(["BAAI/bge-base-en-v1.5"])
    )
    mock_registry_repo.is_document_registered.return_value = mock_doc_record
    SUT = PaperNamer(mock_registry_repo)

    tei_xml_content = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
        <teiHeader>
            <fileDesc>
                <titleStmt>
                    <title>Sample Paper Title</title>
                    <author>John Doe</author>
                </titleStmt>
                <publicationStmt>
                    <date when="2024">2024</date>
                </publicationStmt>
            </fileDesc>
        </teiHeader>
    </TEI>
    """

    tei_xml_path = tmp_path / "sample.tei.xml"
    tei_xml_path.write_text(tei_xml_content, encoding="utf-8")
    result = SUT.resolve_standard_paper_name(ET.fromstring(tei_xml_content), document_hash="hash_existing")

    assert result == "existing_paper"
    mock_registry_repo.is_document_registered.assert_called_once_with(document_hash="hash_existing")

def test_resolve_standard_paper_name_with_stop_words(tmp_path):
    mock_registry_repo = Mock()
    mock_registry_repo.is_document_registered.return_value = None  # Document not yet registered
    mock_registry_repo.is_title_registered.return_value = False
    mock_registry_repo.reserve_title = Mock()
    SUT = PaperNamer(mock_registry_repo)

    tei_xml_content = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
        <teiHeader>
            <fileDesc>
                <titleStmt>
                    <title>The Impact of Using AI in Research</title>
                    <author>Jane Smith</author>
                </titleStmt>
                <publicationStmt>
                    <date when="2023">2023</date>
                </publicationStmt>
            </fileDesc>
        </teiHeader>
    </TEI>
    """

    tei_xml_path = tmp_path / "sample_with_stop_words.tei.xml"
    tei_xml_path.write_text(tei_xml_content, encoding="utf-8")
    standard_name = SUT.resolve_standard_paper_name(ET.fromstring(tei_xml_content), document_hash="hash3")

    assert standard_name == "smith2023research"
    mock_registry_repo.reserve_title.assert_called_once_with("smith2023research")

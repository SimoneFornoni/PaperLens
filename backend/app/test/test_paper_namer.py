import sys
from pathlib import Path
from unittest.mock import Mock

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.paper_namer import PaperNamer

def test_generate_standard_paper_name(tmp_path):
    mock_registry_repo = Mock()
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
    standard_name = SUT.generate_standard_paper_name(tei_xml_path)

    assert standard_name == "doe2024sample"


def test_generate_standard_paper_name_duplicate(tmp_path):
    mock_registry_repo_instance = Mock()
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
    standard_name_1 = SUT.generate_standard_paper_name(tei_xml_path)
    standard_name_2 = SUT.generate_standard_paper_name(tei_xml_path)

    assert standard_name_1 == "doe2024sample"
    assert standard_name_2 == "doe2024sample_1"


def test_generate_standard_paper_name_with_stop_words(tmp_path):
    mock_registry_repo = Mock()
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
    standard_name = SUT.generate_standard_paper_name(tei_xml_path)

    assert standard_name == "smith2023research"

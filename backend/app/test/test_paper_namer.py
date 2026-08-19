import pytest
from unittest.mock import patch
from pathlib import Path

from backend.app.services.paper_namer import PaperNamer

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_generate_standard_paper_name(tmp_path):
    SUT = PaperNamer()

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

    assert standard_name == "doe_2024_sample.pdf"

def test_generate_standard_paper_name_with_stop_words(tmp_path):
    SUT = PaperNamer()

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

    assert standard_name == "smith_2023_research.pdf"

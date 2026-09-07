"""Test cases for the Chunker service."""

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.chunker import Chunker

def test_extract_text_chunks():
    tei_xml_content = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
        <text>
            <body>
                <div>
                    <head>Introduction</head>
                    <p>This is the first paragraph.</p>
                    <p>
                        <s>This is the </s>
                        <s>second one.</s>
                    </p>
                </div>
                <div>
                    <head>Methods</head>
                    <p>This is the methods paragraph.</p>
                </div>
            </body>
        </text>
    </TEI>
    """

    tei_root = ET.fromstring(tei_xml_content)

    SUT = Chunker()

    chunks, full_text = SUT.extract_text_chunks(tei_root)

    # Assertions
    assert len(chunks) == 1
    assert full_text == "This is the first paragraph. This is the second one. This is the methods paragraph."

def test_extract_text_chunks_with_empty_tei():
    tei_xml_content = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
        <text>
            <body>
            </body>
        </text>
    </TEI>
    """

    tei_root = ET.fromstring(tei_xml_content)

    SUT = Chunker()

    chunks, full_text = SUT.extract_text_chunks(tei_root)

    # Assertions
    assert len(chunks) == 0
    assert full_text == ""

def test_extract_text_chunks_with_abstract():
    tei_xml_content = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
        <text>
            <abstract>
                <p>This is the abstract paragraph.</p>
            </abstract>
            <body>
                <div>
                    <head>Introduction</head>
                    <p>This is the first paragraph.</p>
                </div>
            </body>
        </text>
    </TEI>
    """

    tei_root = ET.fromstring(tei_xml_content)

    SUT = Chunker()

    chunks, full_text = SUT.extract_text_chunks(tei_root)

    # Assertions
    assert len(chunks) == 1
    assert full_text == "This is the abstract paragraph. This is the first paragraph."

def test_long_paragraphs():
    tei_xml_content = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
        <text>
            <body>
                <div>
                    <p>This paragraph contains enough words to remain a standalone chunk.</p>
                    <p>Short paragraph here.</p>
                    <p>This paragraph also contains enough words to create another chunk.</p>
                </div>
            </body>
        </text>
    </TEI>
    """

    chunks, full_text = Chunker().extract_text_chunks(ET.fromstring(tei_xml_content))

    assert len(chunks) == 2
    assert chunks[0].text == "This paragraph contains enough words to remain a standalone chunk. Short paragraph here."
    assert full_text == (
        "This paragraph contains enough words to remain a standalone chunk. "
        "Short paragraph here. This paragraph also contains enough words to create another chunk."
    )
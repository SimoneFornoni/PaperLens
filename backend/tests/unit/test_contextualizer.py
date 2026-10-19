import pytest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.indexing.contextualizer import Contextualizer

def test_contextualize():
    mock_model = SimpleNamespace(name="llama3.2:3b")

    def mock_generate(*args, **kwargs):
        prompt = kwargs.get("prompt", "")
        if "<document>" in prompt:
            return {"response": "Sommario del documento"}
        if "<current_chunk>" in prompt:
            return {"response": "Contesto per il chunk"}
        return {"response": ""}

    with patch("app.services.contextualizer.Client") as mock_ollama_client:
        mock_instance = mock_ollama_client.return_value
        mock_instance.models.return_value = [mock_model]
        mock_instance.generate.side_effect = mock_generate

        SUT = Contextualizer(generative_model_name="llama3.2:3b")
        results = SUT.contextualize(
            document_text="Testo del documento", 
            all_texts=["Chunk 1", "Chunk 2"]
        )

    assert len(results) == 2
    assert results[0].original_text == "Chunk 1"
    assert results[0].contextualized_text == "Contesto per il chunk\n\nChunk 1"
    assert results[0].context_text == "Contesto per il chunk"
    assert results[1].original_text == "Chunk 2"
    assert results[1].contextualized_text == "Contesto per il chunk\n\nChunk 2"
    assert results[1].context_text == "Contesto per il chunk"

def test_contextualize_raises_for_missing_model():
    mock_model = SimpleNamespace(name="altro_modello:latest")

    with patch("app.services.contextualizer.Client") as mock_ollama_client:
        mock_instance = mock_ollama_client.return_value
        mock_instance.models.return_value = [mock_model]

        with pytest.raises(ConnectionError, match="Failed to validate connection"):
            Contextualizer(generative_model_name="llama3.2:3b")

def test_contextualize_fallback_on_generate_exception():
    mock_model = SimpleNamespace(name="llama3.2:3b")

    def mock_generate_with_error(*args, **kwargs):
        prompt = kwargs.get("prompt", "")
        if "<document>" in prompt:
            return {"response": "Sommario del documento"}
        raise Exception("Ollama service error")

    with patch("app.services.contextualizer.Client") as mock_ollama_client:
        mock_instance = mock_ollama_client.return_value
        mock_instance.models.return_value = [mock_model]
        mock_instance.generate.side_effect = mock_generate_with_error

        SUT = Contextualizer(generative_model_name="llama3.2:3b")
        results = SUT.contextualize(
            document_text="Testo del documento", 
            all_texts=["Chunk con errore"]
        )

    assert len(results) == 1
    assert results[0].original_text == "Chunk con errore"
    assert results[0].contextualized_text == "Chunk con errore"
    assert results[0].context_text == ""
    
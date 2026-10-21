import pytest
from unittest.mock import patch
from types import SimpleNamespace

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

    with patch("app.services.indexing.contextualizer.Client") as mock_ollama_client:
        mock_instance = mock_ollama_client.return_value
        mock_instance.models.return_value = [mock_model]
        mock_instance.generate.side_effect = mock_generate

        SUT = Contextualizer(generative_model_name="llama3.2:3b")
        results = SUT.contextualize(
            document_text="Testo del documento", 
            all_text_chunks=["Chunk 1", "Chunk 2"]
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

    with patch("app.services.indexing.contextualizer.Client") as mock_ollama_client:
        mock_instance = mock_ollama_client.return_value
        mock_instance.models.return_value = [mock_model]

        with pytest.raises(ConnectionError, match="Failed to validate connection"):
            Contextualizer(generative_model_name="llama3.2:3b")

def test_contextualize_raises_on_generate_exception():
    mock_model = SimpleNamespace(name="llama3.2:3b")

    def mock_generate_with_error(*args, **kwargs):
        prompt = kwargs.get("prompt", "")
        if "<document>" in prompt:
            return {"response": "Sommario del documento"}
        raise RuntimeError("Ollama service error")

    with patch("app.services.indexing.contextualizer.Client") as mock_ollama_client:
        mock_instance = mock_ollama_client.return_value
        mock_instance.models.return_value = [mock_model]
        mock_instance.generate.side_effect = mock_generate_with_error

        SUT = Contextualizer(generative_model_name="llama3.2:3b")
        with pytest.raises(RuntimeError, match="Ollama service error"):
            SUT.contextualize(
                document_text="Testo del documento",
                all_text_chunks=["Chunk con errore"]
            )


def test_contextualize_raises_on_summary_generation_exception():
    mock_model = SimpleNamespace(name="llama3.2:3b")

    with patch("app.services.indexing.contextualizer.Client") as mock_ollama_client:
        mock_instance = mock_ollama_client.return_value
        mock_instance.models.return_value = [mock_model]
        mock_instance.generate.side_effect = RuntimeError("Ollama summary error")

        SUT = Contextualizer(generative_model_name="llama3.2:3b")
        with pytest.raises(RuntimeError, match="Ollama summary error"):
            SUT.contextualize(
                document_text="Testo del documento",
                all_text_chunks=["Chunk 1"]
            )


def test_contextualize_raises_on_empty_context_response():
    mock_model = SimpleNamespace(name="llama3.2:3b")

    def mock_generate(*args, **kwargs):
        if "<document>" in kwargs.get("prompt", ""):
            return {"response": "Sommario del documento"}
        return {"response": ""}

    with patch("app.services.indexing.contextualizer.Client") as mock_ollama_client:
        mock_instance = mock_ollama_client.return_value
        mock_instance.models.return_value = [mock_model]
        mock_instance.generate.side_effect = mock_generate

        SUT = Contextualizer(generative_model_name="llama3.2:3b")
        with pytest.raises(RuntimeError, match="Context generation returned an empty response"):
            SUT.contextualize(
                document_text="Testo del documento",
                all_text_chunks=["Chunk 1"]
            )


def test_contextualize_raises_on_empty_summary_response():
    mock_model = SimpleNamespace(name="llama3.2:3b")

    with patch("app.services.indexing.contextualizer.Client") as mock_ollama_client:
        mock_instance = mock_ollama_client.return_value
        mock_instance.models.return_value = [mock_model]
        mock_instance.generate.return_value = {"response": ""}

        SUT = Contextualizer(generative_model_name="llama3.2:3b")
        with pytest.raises(RuntimeError, match="Document summary generation returned an empty response"):
            SUT.contextualize(
                document_text="Testo del documento",
                all_text_chunks=["Chunk 1"]
            )
    
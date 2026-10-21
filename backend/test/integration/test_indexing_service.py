from types import SimpleNamespace
from unittest.mock import Mock

from app.services.indexing import indexing_service as indexing_module
from app.services.indexing.indexing_service import IndexingService


def configure_pipeline(monkeypatch, ingestion_status="ingested", contextualization_enabled=False):
    connection = Mock()
    registry_repo = Mock()
    monkeypatch.setattr(indexing_module, "QdrantConnectionProvider", Mock(return_value=connection))
    monkeypatch.setattr(indexing_module, "RegistryRepository", Mock(return_value=registry_repo))

    ingestor = Mock()
    ingestor.ingest_document.return_value = {
        "status": ingestion_status,
        "document_hash": "document-hash",
        "title": "existing-title" if ingestion_status == "already_indexed" else None,
    }
    monkeypatch.setattr(indexing_module, "DocumentIngestor", Mock(return_value=ingestor))

    parser = Mock()
    parser.parse_pdf.return_value = object()
    monkeypatch.setattr(indexing_module, "PDFParser", Mock(return_value=parser))

    paper_namer = Mock()
    paper_namer.resolve_standard_paper_name.return_value = "paper-title"
    monkeypatch.setattr(indexing_module, "PaperNamer", Mock(return_value=paper_namer))

    chunk = SimpleNamespace(
        text="chunk text",
        section="body",
        page_num=1,
        chunk_idx=0,
        sentences=[],
    )
    chunker = Mock()
    chunker.extract_text_chunks.return_value = ([chunk], "document text")
    monkeypatch.setattr(indexing_module, "Chunker", Mock(return_value=chunker))

    embedder = Mock()
    embedder.dense_vector_size = 3
    embedder.embed_text.return_value = {"dense": [[0.1, 0.2, 0.3]], "sparse": [{"indices": [1], "values": [0.5]}]}
    monkeypatch.setattr(indexing_module, "Embedder", Mock(return_value=embedder))

    vector_repo = Mock()
    vector_repo.upsert_chunks.return_value = 1
    monkeypatch.setattr(indexing_module, "VectorRepository", Mock(return_value=vector_repo))

    service = IndexingService(
        output_dir=None,
        document_path=None,
        contextualization_enabled=contextualization_enabled,
    )
    return service, parser, registry_repo, vector_repo


def test_pipeline_returns_indexed_result(monkeypatch):
    service, _, registry_repo, vector_repo = configure_pipeline(monkeypatch)

    result = service.run_indexing_pipeline()

    assert result.status == "indexed"
    assert result.succeeded
    assert result.document_hash == "document-hash"
    assert result.title == "paper-title"
    assert result.chunk_count == 1
    registry_repo.register_document.assert_called_once_with(
        document_hash="document-hash",
        title="paper-title",
        model="BAAI/bge-base-en-v1.5",
    )
    vector_repo.delete_document_chunks.assert_not_called()


def test_pipeline_skips_already_indexed_document(monkeypatch):
    service, parser, registry_repo, vector_repo = configure_pipeline(
        monkeypatch,
        ingestion_status="already_indexed",
    )

    result = service.run_indexing_pipeline()

    assert result.status == "already_indexed"
    assert result.succeeded
    assert result.title == "existing-title"
    parser.parse_pdf.assert_not_called()
    registry_repo.register_document.assert_not_called()
    vector_repo.upsert_chunks.assert_not_called()


def test_pipeline_returns_error_when_enabled_contextualization_fails(monkeypatch):
    service, _, registry_repo, vector_repo = configure_pipeline(
        monkeypatch,
        contextualization_enabled=True,
    )
    contextualizer = Mock()
    contextualizer.contextualize.side_effect = RuntimeError("Ollama service error")
    monkeypatch.setattr(indexing_module, "Contextualizer", Mock(return_value=contextualizer))

    result = service.run_indexing_pipeline()

    assert result.status == "failed"
    assert not result.succeeded
    assert result.error == "Ollama service error"
    registry_repo.register_document.assert_not_called()
    vector_repo.upsert_chunks.assert_not_called()
    vector_repo.delete_document_chunks.assert_not_called()


def test_pipeline_removes_partial_vectors_after_upsert_failure(monkeypatch):
    service, _, registry_repo, vector_repo = configure_pipeline(monkeypatch)
    vector_repo.upsert_chunks.side_effect = RuntimeError("Qdrant write failed")

    result = service.run_indexing_pipeline()

    assert result.status == "failed"
    assert not result.succeeded
    assert result.error == "Qdrant write failed"
    assert result.rollback_errors == ()
    vector_repo.delete_document_chunks.assert_called_once_with("document-hash")
    registry_repo.unregister_document_model.assert_not_called()
    registry_repo.register_document.assert_not_called()


def test_pipeline_rolls_back_vectors_and_registry_after_registry_failure(monkeypatch):
    service, _, registry_repo, vector_repo = configure_pipeline(monkeypatch)
    registry_repo.register_document.side_effect = RuntimeError("Registry write failed")

    result = service.run_indexing_pipeline()

    assert result.status == "failed"
    assert result.error == "Registry write failed"
    registry_repo.unregister_document_model.assert_called_once_with(
        "document-hash",
        "BAAI/bge-base-en-v1.5",
    )
    vector_repo.delete_document_chunks.assert_called_once_with("document-hash")


def test_pipeline_reports_cleanup_failures(monkeypatch):
    service, _, registry_repo, vector_repo = configure_pipeline(monkeypatch)
    registry_repo.register_document.side_effect = RuntimeError("Registry write failed")
    registry_repo.unregister_document_model.side_effect = RuntimeError("Registry cleanup failed")
    vector_repo.delete_document_chunks.side_effect = RuntimeError("Vector cleanup failed")

    result = service.run_indexing_pipeline()

    assert result.status == "failed"
    assert result.error == "Registry write failed"
    assert result.rollback_errors == (
        "Registry cleanup failed: Registry cleanup failed",
        "Vector cleanup failed: Vector cleanup failed",
    )

"""Class responsible of executing the indexing pipeline on given documents."""

import logging
from pathlib import Path

from app.repositories.qdrant_connection_provider import QdrantConnectionProvider
from app.repositories.registry_repository import RegistryRepository
from app.services.indexing.document_ingestor import DocumentIngestor
from app.services.indexing.pdf_parser import PDFParser
from app.services.indexing.chunker import Chunker
from app.services.indexing.contextualizer import Contextualizer
from app.services.indexing.embedder import Embedder
from app.services.indexing.paper_namer import PaperNamer
from app.repositories.vector_repository import VectorRepository

from app.models.domain import Chunk
from app.services.indexing.models import ContextualizedChunk, IndexingResult

logger = logging.getLogger(__name__)

class IndexingService:
    def __init__(self, 
                output_dir: Path,
                document_path: Path,
                dense_model_name: str = "BAAI/bge-base-en-v1.5",
                contextualization_enabled: bool = False,
                qdrant_url: str = "http://localhost:6333",
                grobid_url: str = "http://localhost:8070",
                ollama_url: str = "http://localhost:11434"):
        """Initialize the IndexingService.
        
        Args:
            output_dir (Path): The directory where intermediate files will be stored.
            document_path (Path): The path to the document to be indexed.
            dense_model_name (str): The name of the dense embedding model to use.
            contextualization_enabled (bool): Whether to enable contextualization of text chunks.
            qdrant_url (str): The URL of the Qdrant service.
            grobid_url (str): The URL of the GROBID service.
            ollama_url (str): The URL of the Ollama service.
        """

        self.output_dir = output_dir
        self.document_path = document_path
        self.dense_model_name = dense_model_name
        self.contextualization_enabled = contextualization_enabled
        self.qdrant_url = qdrant_url
        self.grobid_url = grobid_url
        self.ollama_url = ollama_url

    def run_indexing_pipeline(self) -> IndexingResult:
        """Run indexing pipeline, doing a rollback of Qdrant writes if a phase fails.
        
        Returns:
            IndexingResult: The result of the indexing operation, including status and any errors.
        """
        document_hash = None
        standard_name = None
        registry_repo = None
        vector_repo = None
        vector_write_started = False
        registry_write_started = False

        try:
            qdrant_connection = QdrantConnectionProvider(qdrant_url=self.qdrant_url)
            registry_repo = RegistryRepository(connection=qdrant_connection)
    
            document_ingestor = DocumentIngestor(
                registry_repo=registry_repo, 
                output_dir=self.output_dir, 
                dense_model_name=self.dense_model_name
            )
            ingestion_result = document_ingestor.ingest_document(self.document_path)
            document_hash = ingestion_result["document_hash"]
            if ingestion_result["status"] == "already_indexed":
                logger.info("Document '%s' is already indexed for model '%s'.", document_hash, self.dense_model_name)
                return IndexingResult(
                    status="already_indexed",
                    document_hash=document_hash,
                    title=ingestion_result.get("title"),
                )
    
            grobid_manager = PDFParser(grobid_url=self.grobid_url, output_dir=self.output_dir)
            et_element = grobid_manager.parse_pdf(self.document_path)
            if et_element is None:
                raise RuntimeError("GROBID did not produce TEI XML for the document.")
    
            paper_namer = PaperNamer(registry_repo=registry_repo)
            standard_name = paper_namer.resolve_standard_paper_name(et_element, document_hash=document_hash)
    
            chunker = Chunker()
            text_chunks, full_document_text = chunker.extract_text_chunks(et_element)
            if not text_chunks:
                raise ValueError("No text chunks were extracted from the document.")
    
            if self.contextualization_enabled:
                all_text_chunks = [chunk.text for chunk in text_chunks]
                contextualizer = Contextualizer(ollama_url=self.ollama_url)
                contextualized_chunks = contextualizer.contextualize(all_text_chunks, full_document_text)
            else:
                contextualized_chunks = [
                    ContextualizedChunk(
                        original_text=chunk.text,
                        contextualized_text=chunk.text,
                        context_text="",
                    )
                    for chunk in text_chunks
                ]

            if len(text_chunks) != len(contextualized_chunks):
                raise ValueError(
                    "The number of contextualized chunks does not match the extracted chunks."
                )
    
            texts_to_embed = [chunk.contextualized_text for chunk in contextualized_chunks]
            embedder = Embedder(dense_model_name=self.dense_model_name)
            embeddings = embedder.embed_text(texts_to_embed)
    
            chunks = []
            for text_chunk, contextualized_chunk in zip(text_chunks, contextualized_chunks):
                chunk = Chunk(
                    text=contextualized_chunk.original_text,
                    context_text=contextualized_chunk.context_text,
                    title=standard_name,
                    document_hash=document_hash,
                    section=text_chunk.section,
                    page_num=text_chunk.page_num,
                    chunk_idx=text_chunk.chunk_idx,
                    sentence_count=len(text_chunk.sentences),
                    sentences=text_chunk.sentences
                )
                chunks.append(chunk)
    
            vector_repo = VectorRepository(
                connection=qdrant_connection, 
                dense_model_name=self.dense_model_name, 
                dense_vector_size=embedder.dense_vector_size
            )
            vector_write_started = True
            upserted_count = vector_repo.upsert_chunks(chunks, embeddings["dense"], embeddings["sparse"])
            if upserted_count != len(chunks):
                raise RuntimeError(
                    f"Expected to upsert {len(chunks)} chunks, but upserted {upserted_count}."
                )

            registry_write_started = True
            registry_repo.register_document(
                document_hash=document_hash,
                title=standard_name,
                model=self.dense_model_name,
            )

            logger.info(
                "Indexed document '%s' as '%s' with %d chunks.",
                document_hash,
                standard_name,
                upserted_count,
            )

            return IndexingResult(
                status="indexed",
                document_hash=document_hash,
                title=standard_name,
                chunk_count=upserted_count,
            )
        except Exception as error:
            logger.exception("Indexing failed for document '%s'.", document_hash or self.document_path)
            rollback_errors = []

            if registry_write_started and registry_repo is not None and document_hash is not None:
                try:
                    registry_repo.unregister_document_model(document_hash, self.dense_model_name)
                except Exception as rollback_error:
                    rollback_errors.append(f"Registry cleanup failed: {rollback_error}")
                    logger.exception("Failed to roll back registry entry for '%s'.", document_hash)

            if vector_write_started and vector_repo is not None and document_hash is not None:
                try:
                    vector_repo.delete_document_chunks(document_hash)
                except Exception as rollback_error:
                    rollback_errors.append(f"Vector cleanup failed: {rollback_error}")
                    logger.exception("Failed to roll back vector points for '%s'.", document_hash)

            return IndexingResult(
                status="failed",
                document_hash=document_hash,
                title=standard_name,
                error=str(error),
                rollback_errors=tuple(rollback_errors),
            )
        
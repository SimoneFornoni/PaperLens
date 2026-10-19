from pathlib import Path

from app.repositories.qdrant_connection_provider import QdrantConnectionProvider
from app.repositories.registry_repository import RegistryRepository
from app.services.indexing.document_ingestor import DocumentIngestor
from app.services.indexing.pdf_parser import GROBIDManager
from app.services.indexing.chunker import Chunker
from app.services.indexing.contextualizer import Contextualizer
from app.services.indexing.embedder import Embedder
from app.services.indexing.paper_namer import PaperNamer
from app.repositories.vector_repository import VectorRepository

from app.models.domain import Chunk, ContextualizedChunk

# TODO: Add rollback mechanism
# TODO: Non sono completamente sicuro del riuso del file tei in caso ci sia già, capire se l'ho fatto
# FIXME: Lo sparse vector viene sempre salvato vuoto... non so perché
def main():
    output_dir = Path("C:/Progetti Git/PaperLens/output")
    document_path = Path("C:/Progetti Git/Tesi/articles/Test/RAG_Advanced.pdf")
    dense_model_name = "BAAI/bge-base-en-v1.5"
    contextualization_enabled = False

    qdrant_url = "http://localhost:6333"
    grobid_url = "http://localhost:8070"

    try:
        qdrant_connection = QdrantConnectionProvider(qdrant_url=qdrant_url)
        registry_repo = RegistryRepository(connection=qdrant_connection)

        document_ingestor = DocumentIngestor(
            registry_repo=registry_repo, 
            output_dir=output_dir, 
            dense_model_name=dense_model_name
        )
        result = document_ingestor.ingest_document(document_path)
        print(f"Ingestion result: {result}")

        grobid_manager = GROBIDManager(grobid_url=grobid_url, output_dir=output_dir)
        et_element = grobid_manager.parse_pdf(document_path)
        print(f"TEI XML generated at: {et_element}")

        paper_namer = PaperNamer(registry_repo=registry_repo)
        standard_name = paper_namer.resolve_standard_paper_name(et_element, document_hash=result.get("document_hash"))
        print(f"Standard paper name: {standard_name}")

        chunker = Chunker()
        text_chunks, full_text = chunker.extract_text_chunks(et_element)
        print(f"Number of chunks created: {len(text_chunks)}")
        #print(f"Chunks: {text_chunks[:2]}")

        if contextualization_enabled:
            all_texts = [chunk.text for chunk in text_chunks]
            contextualizer = Contextualizer()
            contextualized_chunks = contextualizer.contextualize(all_texts, full_text)
            print(f"Number of contextualized chunks: {len(contextualized_chunks)}")
            #print(f"Contextualized Chunks: {contextualized_chunks[:2]}")
        else:
            contextualized_chunks = [
                ContextualizedChunk(
                    original_text=chunk.text,
                    contextualized_text=chunk.text,
                    context_text="",
                )
                for chunk in text_chunks
            ]

        all_context_texts = [chunk.context_text for chunk in contextualized_chunks]
        embedder = Embedder(dense_model_name=dense_model_name)
        embeddings = embedder.embed_text(all_context_texts)
        print(f"Number of embeddings created: {len(embeddings)}")
        #print(f"Embeddings: {embeddings[:2]}")

        chunks = []
        for text_chunk, contextualized_chunk in zip(text_chunks, contextualized_chunks):
            chunk = Chunk(
                text=contextualized_chunk.original_text,
                context_text=contextualized_chunk.context_text,
                title=standard_name,
                document_hash=result.get("document_hash"),
                section=text_chunk.section,
                page_num=text_chunk.page_num,
                chunk_idx=text_chunk.chunk_idx,
                sentence_count=len(text_chunk.sentences),
                sentences=text_chunk.sentences
            )
            chunks.append(chunk)

        vector_repo = VectorRepository(connection=qdrant_connection)
        upserted_count = vector_repo.upsert_chunks(chunks, embeddings["dense"], embeddings["sparse"])
        print(f"Number of chunks upserted into Qdrant: {upserted_count}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
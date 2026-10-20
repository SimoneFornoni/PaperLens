from pathlib import Path

from app.services.indexing.indexing_service import IndexingService

# TODO: fare indexing service
# TODO: Add rollback mechanism
# TODO: Non sono completamente sicuro del riuso del file tei in caso ci sia già, capire se l'ho fatto
def main():
    output_dir = Path("C:/Progetti Git/PaperLens/output")
    document_path = Path("C:/Progetti Git/Tesi/articles/Test/RAG_Advanced.pdf")
    dense_model_name = "sentence-transformers/all-MiniLM-L6-v2" #BAAI/bge-base-en-v1.5
    contextualization_enabled = False

    qdrant_url = "http://localhost:6333"
    grobid_url = "http://localhost:8070"

    indexing_service = IndexingService(
        output_dir=output_dir,
        document_path=document_path,
        dense_model_name=dense_model_name,
        contextualization_enabled=contextualization_enabled,
        qdrant_url=qdrant_url,
        grobid_url=grobid_url
    )

    indexing_service.run_indexing_pipeline()

if __name__ == "__main__":
    main()
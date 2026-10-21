from pathlib import Path

from app.services.indexing.indexing_service import IndexingService

#TODO: generali
#   index di un singolo pdf alla volta.
#   (si potrebbe permettere di stoppare e poi riprendere il processo: https://github.com/FornoniSimone/zoteroRAG-Ffork/commit/bfc6d65194626938d695c39c2323d7816d690e76)
#   scegliere un sottoinsieme degli articoli su cui fare la ricerca
#   spostare il la cartella locale di Qdrant in output??? Forse non si può fare

# TODO: Indexing:
#   Rollback anche della cache + Consistency Check

# TODO: Retrieval:
#   Il tipo ha cambiato molto l'evidenziazione, forse era molto inefficiente?? capiamo
#   Exact Search su Qdrant
#   Il bro usa come reranking gte-reranker-modernbert-base

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

    result = indexing_service.run_indexing_pipeline()
    if not result.succeeded:
        raise RuntimeError(result.error or "Document indexing failed.")

if __name__ == "__main__":
    main()
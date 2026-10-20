# PaperLens
Question-answering system for scientific papers with a RAG pipeline: indexing → retrieval → extractive QA.

# Run

```powershell
cd backend
uv run pytest
uv run python -m app.main
```

## Resume
The system developed is a Retrieval-Augmented Generation (RAG) application designed to support scientific literature searches. The main objective is not to generate an answer directly, but to precisely identify the paragraphs in scientific articles that contain the information requested by the user, highlighting the sentences deemed most relevant within the documents. In this way, the system keeps the researcher at the center of the decision-making process, providing both the extracted answer and the original context from which it comes.

The architecture is divided into two main phases: indexing and retrieval. During indexing, scientific articles in PDF format are processed using GROBID, which extracts the document structure and paragraphs. The paragraphs are then divided into chunks, optionally contextualized using a Large Language Model, transformed into embeddings with a configurable embedding model, and stored in the Qdrant vector database. In addition to the embeddings, the metadata necessary to reconstruct the position of the paragraphs in the original documents is also saved.

During the retrieval phase, the user enters a question in natural language. The system automatically generates several paraphrases of the query to improve information retrieval and searches for the most relevant chunks using a vector similarity search on Qdrant. The results are then refined by a reranker, which reorders the retrieved documents based on their actual relevance to the question. Finally, an extractive question-answering model identifies the sentences containing the answer within the selected chunks; these sentences are highlighted directly in the PDFs, allowing the user to quickly verify the context of the answer.

The main technologies used include Python for application development, with a Streamlit web interface. Qdrant as a vector database for storing and searching embeddings, GROBID for extracting the structure of scientific documents, embedding models for the vector representation of texts, a reranker to improve retrieval quality, and Large Language Model (LLM) models for both contextualizing chunks during indexing and generating question variants during retrieval. The entire system is designed as a modular pipeline, in which each component can be configured or replaced independently of the others.
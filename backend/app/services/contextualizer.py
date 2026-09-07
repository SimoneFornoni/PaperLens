"""Contextualizer Service"""
import logging
from typing import List
from ollama import Client

from app.models.domain import ContextualizedChunk

logger = logging.getLogger(__name__)

class Contextualizer:
    """Handles the contextualization of text data using a generative model via ollama."""

    _CHARS_PER_TOKEN: int = 4
    _BUFFER_TOKENS: int = 250
    _DEFAULT_MIN_CTX: int = 2048
    _DEFAULT_MAX_CTX: int = 16384
    _PREDICT_MAX_TOKENS_CHUNK: int = 100
    _PREDICT_MAX_TOKENS_SUMMARY: int = 400

    def __init__(self,
                ollama_url: str = "http://localhost:11434",
                generative_model_name: str = "llama3.2:3b"):
        """Initialize the Contextualizer with specified parameters.

        Args:
            ollama_url (str): The URL of the ollama service.
            generative_model_name (str): The name of the generative model to use for contextualization.
        """
        self.ollama_url = ollama_url
        self.generative_model_name = generative_model_name

        self.ollama_client = self._initialize_ollama_client()
        self._validate_ollama_connection()
        logger.info(f"Initialized Ollama client with model '{self.generative_model_name}' at URL: {self.ollama_url}")

    def _initialize_ollama_client(self) -> Client:
        """Initialize the ollama client.
        
        Returns:
            Client: An instance of the ollama Client.
        """
        try:
            return Client(host=self.ollama_url)
        except Exception as e:
            logger.error(f"Failed to connect to ollama service at {self.ollama_url}: {e}")
            raise ConnectionError(f"Failed to connect to ollama service at {self.ollama_url}: {e}")

    def _validate_ollama_connection(self):
        """Validate the connection to the ollama service."""
        try:
            models = self.ollama_client.models()
            if self.generative_model_name not in [model.name for model in models]:
                logger.error(f"Generative model '{self.generative_model_name}' not found in ollama service.")
                raise ValueError(f"Generative model '{self.generative_model_name}' not found in ollama service.")
        except Exception as e:
            logger.error(f"Failed to validate connection to ollama service at {self.ollama_url}: {e}")
            raise ConnectionError(f"Failed to validate connection to ollama service at {self.ollama_url}: {e}")
    
    def contextualize(self, 
                      all_texts: list[str],
                      document_text: str) -> List[ContextualizedChunk]:
        """Generate contextualized chunks of text by prompting an LLM to provide context for each chunk.
        
        Args:
            document_text (str): The full text of the document.
            all_texts (list[str]): A list of text chunks to contextualize.
            
        Returns:
            List[ContextualizedChunk]: The original, contextualized, and
                context-only text for each chunk.
        """
        contextualized_results = []

        doc_summary = self._generate_document_summary(document_text)
        optimal_num_ctx = self._determine_optimal_context_length(all_texts, len(doc_summary))

        prompt_template = (
            "<document_summary>\n{doc_summary}\n</document_summary>\n\n"
            "<current_chunk>\n{chunk}\n</current_chunk>\n\n"
            "You are an assistant organizing a vector database. Based on the global summary above, "
            "provide a succinct context (max 30 words) that describes what THIS chunk is about, "
            "so that it can be found via semantic search.\n"
            "Rules:\n"
            "- Name the specific topic, entity, method, or metric discussed in this chunk.\n"
            "- Avoid generic phrases like 'This chunk discusses...' or 'In this section...'.\n"
            "- Output only the context sentence, without labels, bullets, or quotes.\n\n"
            "Context sentence:"
        )

        for chunk in all_texts:
            formatted_prompt = prompt_template.format(
                doc_summary=doc_summary, 
                chunk=chunk.strip()
            )

            try:
                response = self.ollama_client.generate(
                    model=self.generative_model_name,
                    prompt=formatted_prompt,
                    options={
                        "temperature": 0.1, 
                        "num_predict": self._PREDICT_MAX_TOKENS_CHUNK,
                        "num_ctx": optimal_num_ctx
                    },
                    keep_alive=-1
                )

                context_prefix = response["response"].strip().strip('"').strip("'")
                context_prefix = context_prefix.replace("```", "").strip()

                full_chunk = f"{context_prefix}\n\n{chunk}" if context_prefix else chunk
                contextualized_results.append(ContextualizedChunk(
                    original_text=chunk,
                    contextualized_text=full_chunk,
                    context_text=context_prefix,
                ))
            except Exception as e:
                logger.error(f"Error during contextualization for chunk: '{chunk[:10]}...': {e}")
                contextualized_results.append(ContextualizedChunk(
                    original_text=chunk,
                    contextualized_text=chunk,
                    context_text="",
                ))

        logger.info(f"Contextualization completed for {len(all_texts)} chunks.")
        self._flush_ollama_cache()
        return contextualized_results

    def _determine_optimal_context_length(self, all_texts: list[str], doc_summary_length: int) -> int:
            """Determine the optimal context length for the LLM based on the document summary and text chunks.
    
            Args:
                all_texts (list[str]): A list of text chunks to contextualize.
                doc_summary_length (int): The length of the document summary.
    
            Returns:
                int: The optimal context length for the LLM.
            """
            max_chunk_chars = max((len(chunk) for chunk in all_texts), default=0)
            prompt_tokens = (doc_summary_length + max_chunk_chars) // self._CHARS_PER_TOKEN
            required_ctx = prompt_tokens + self._PREDICT_MAX_TOKENS_CHUNK + self._BUFFER_TOKENS
            return max(self._DEFAULT_MIN_CTX, min(required_ctx, self._DEFAULT_MAX_CTX)) 
    
    def _generate_document_summary(self, document_text: str) -> str:
        """Generates a concise summary that captures the main themes and content of the given document, using a generative model.
        
        Args:
            document_text (str): The full text of the document.
            
        Returns:
            str: A concise summary of the document.
        """
        optimal_num_ctx = self._determine_optimal_context_length_for_summary(len(document_text))

        prompt = (
            "You are an expert AI assistant preparing metadata for a vector database.\n"
            "Read the following document and write a dense, entity-rich summary in exactly one paragraph.\n"
            "Include the main subject and scope, specific systems, methods, algorithms, datasets, metrics,\n"
            "and key entities such as organizations, authors, products, standards, and technologies.\n"
            "Do NOT use bullets, numbered lists, or markdown formatting.\n\n"
            f"<document>\n{document_text}\n</document>"
        )

        try:
            response = self.ollama_client.generate(
                model=self.generative_model_name,
                prompt=prompt,
                options={
                    "temperature": 0.1, 
                    "num_predict": self._PREDICT_MAX_TOKENS_SUMMARY,
                    "num_ctx": optimal_num_ctx
                },
                keep_alive=-1
            )

            logger.info("Document summary generated successfully.")
            return response['response'].strip()
        except Exception as e:
            logger.warning(f"Document summary generation failed: {e}")
            return ""

    def _determine_optimal_context_length_for_summary(self, document_text_length: int) -> int:
        """Determine the optimal context length for the LLM to generate a summary of the document
            based on document text length.
            
        Args:
            document_text_length (int): The length of the document text.

        Returns:
            int: The optimal context length for the LLM.
        """
        required_ctx = (document_text_length // self._CHARS_PER_TOKEN) + self._PREDICT_MAX_TOKENS_SUMMARY + self._BUFFER_TOKENS
        return max(self._DEFAULT_MIN_CTX, min(required_ctx, self._DEFAULT_MAX_CTX))

    def _flush_ollama_cache(self):
        """Flush Ollama's model cache to free up memory."""
        try:
            self.ollama_client.generate(model=self.generative_model_name, keep_alive=0)
            logger.info("Successfully flushed Ollama cache for model: %s", self.generative_model_name)
        except Exception as e:
            logger.warning(f"Failed to flush ollama cache: {e}")

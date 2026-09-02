"""Module for managing the registry collection in Qdrant."""
import logging
import uuid
from typing import List, Tuple

from backend.app.services.qdrant_connection_provider import QdrantConnectionProvider
from qdrant_client.http import models as qm

logger = logging.getLogger(__name__)

class RegistryRepository:
    """Manages the registry collection in Qdrant, allowing for document registration and retrieval."""

    def __init__(self, connection: QdrantConnectionProvider):
        """Initializes the RegistryRepository with a Qdrant connection provider.
        
        Args:
            connection (QdrantConnectionProvider): The connection provider for Qdrant.
        """
        self._connection = connection
        self.collection_name = "PaperLens_Registry"
        self.reserved_titles: List[str] = []
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensures that the registry collection exists in Qdrant, creating it if necessary."""
        if not self._connection.client.collection_exists(self.collection_name):
            self._connection.create_collection(
                collection_name=self.collection_name,
                vectors_config=None
            )

            self._connection.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="title",
                field_schema=qm.KeywordIndexParams(
                    type="keyword",
                    enable_hnsw=False
                )
            )

            self._connection.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="models",
                field_schema=qm.KeywordIndexParams(
                    type="keyword",
                    enable_hnsw=False
                )
            )

            logger.info(f"Qdrant registry collection '{self.collection_name}' is ready.")

    def _generate_point_id(self, document_hash: str) -> str:
        """Generates a unique point ID for Qdrant.
        
        Args:
            document_hash (str): The hash of the document.
            
        Returns:
            str: A deterministic UUID string based on the document hash.
        """
        input_str = f"{document_hash}"
        NAMESAPCE_PaperLens = uuid.UUID("12345678-1234-5678-1234-567812345678")
        return str(uuid.uuid5(NAMESAPCE_PaperLens, input_str))

    def register_document(self, document_hash: str, title: str, model: str):
        """Registers a document in the registry collection.
        
        Args:
            document_hash (str): The hash of the document.
            title (str): The title of the document.
            model (str): The embedding model used for the document indexing.
        """
        if not document_hash or not title or not model:
            raise ValueError("Document hash, title, and model must be provided.")

        point_id = self._generate_point_id(document_hash)
        existing = self._connection.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=False
        )

        if existing and existing[0].payload:
            existing_payload = existing[0].payload
            existing_models = existing_payload.get("models", [])

            if model not in existing_models:
                existing_models.append(model)
                updated_payload = {
                    "doucument_hash": document_hash,
                    "title": title,
                    "models": existing_models
                }

                self._connection.client.set_payload(
                    collection_name=self.collection_name,
                    points=[point_id],
                    payload=updated_payload
                )
            else:
                logger.warning(f"Document with hash '{document_hash}' is already registered with model '{model}'.")
        else:
            payload = {
                "doucument_hash": document_hash,
                "title": title,
                "models": [model]
            }

            self._connection.client.upsert(
                collection_name=self.collection_name,
                points=[
                    qm.PointStruct(
                        id=point_id,
                        vector={},
                        payload=payload
                    )
                ]
            )

        if title in self.reserved_titles:
            self.reserved_titles.remove(title)

    def is_document_registered(self, document_hash: str, model: str) -> Tuple[str, bool]:
        """Checks if a document is registered in the registry collection for a specific model.
        
        Args:
            document_hash (str): The hash of the document.
            model (str): The embedding model to check against.
        
        Returns:
            Tuple[str, bool]: A tuple containing the title of the document and a boolean indicating if it is registered.
        """
        registry_id = self._generate_point_id(document_hash)
        flt = qm.Filter(
            must=[
                qm.HasIdCondition(has_id=[registry_id]),
                qm.FieldCondition(
                    key="models",
                    match=qm.MatchValue(value=model)
                )
            ]
        )

        try:
            next_offset = 0
            while True:
                points, next_offset = self._connection.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=flt,
                    limit=1,
                    with_payload=["title"],
                    with_vectors=False,
                    offset=next_offset
                )

                if points:
                    return points[0].payload.get("title", ""), True

                if next_offset is None:
                    break
        except Exception as e:
            logger.error(f"Error while checking if document is indexed: {e}")
            raise Exception(f"Error while checking if document is indexed: {e}")
        return "", False

    def is_title_registered(self, title: str) -> bool:
        """Checks if a title is already registered in the whole registry collection.
        
        Args:
            title (str): The title to check for registration.
            
        Returns:
            bool: True if the title is registered, False otherwise.
        """
        flt = qm.Filter(
            must=[
                qm.FieldCondition(
                    key="title",
                    match=qm.MatchValue(value=title)
                )
            ]
        )

        try:
            next_offset = 0
            while True:
                points, next_offset = self._connection.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=flt,
                    limit=1,
                    with_payload=False,
                    with_vectors=False,
                    offset=next_offset
                )

                if points:
                    return True

                if next_offset is None:
                    break
        except Exception as e:
            logger.error(f"Error while checking if title is used: {e}")
            raise Exception(f"Error while checking if title is used: {e}")
        return False

    def reserve_title(self, title: str):
        """Reserves a title in the registry collection to prevent future conflicts.
        
        Args:
            title (str): The title to reserve.
        """
        if not title:
            raise ValueError("Title must be provided.")

        if self.is_title_registered(title):
            raise ValueError(f"Title '{title}' is already registered.")

        self.reserved_titles.append(title)

    def list_registered_documents(self, model: str) -> List[str]:
        """Lists all registered document titles for a specific model in the registry collection.
        
        Args:
            model (str): The embedding model to filter registered documents.
            
        Returns:
            List[str]: A list of registered document titles for the specified model.
        """
        flt = qm.Filter(
            must=[
                qm.FieldCondition(
                    key="models",
                    match=qm.MatchValue(value=model)
                )
            ]
        )

        registered_titles = []
        try:
            next_offset = 0
            while True:
                points, next_offset = self._connection.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=flt,
                    limit=128,
                    with_payload=["title"],
                    with_vectors=False,
                    offset=next_offset
                )

                for point in points:
                    title = point.payload.get("title")
                    if title:
                        registered_titles.append(title)

                if next_offset is None:
                    break
        except Exception as e:
            logger.error(f"Error while listing indexed documents: {e}")
            raise Exception(f"Error while listing indexed documents: {e}")
        return registered_titles
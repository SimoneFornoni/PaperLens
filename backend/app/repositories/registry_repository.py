"""Module for managing the registry collection in Qdrant."""
import logging
import uuid
from typing import List

from app.repositories.qdrant_connection_provider import QdrantConnectionProvider
from app.repositories.models import RegistryEntry
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
            self._connection.client.create_collection(
                collection_name=self.collection_name,
                hnsw_config=qm.HnswConfigDiff(
                    m=0
                ),
                vectors_config=None
            )

            self._connection.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="title",
                field_schema=qm.KeywordIndexParams(
                    type="keyword",
                )
            )

            self._connection.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="models",
                field_schema=qm.KeywordIndexParams(
                    type="keyword",
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
                    "document_hash": document_hash,
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
                "document_hash": document_hash,
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

    def unregister_document_model(self, document_hash: str, model: str) -> None:
        """Remove one model from a document's registry entry, preserving other models.
        
        Args:
            document_hash (str): The hash of the document.
            model (str): The embedding model to remove from the document's registry entry.
        """
        point_id = self._generate_point_id(document_hash)
        existing = self._connection.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )

        if not existing or not existing[0].payload:
            return

        payload = existing[0].payload
        models = list(payload.get("models", []))
        if model not in models:
            return

        remaining_models = [registered_model for registered_model in models if registered_model != model]
        if remaining_models:
            self._connection.client.set_payload(
                collection_name=self.collection_name,
                points=[point_id],
                payload={
                    "document_hash": document_hash,
                    "title": payload.get("title", ""),
                    "models": remaining_models,
                },
            )
        else:
            self._connection.client.delete(
                collection_name=self.collection_name,
                points_selector=qm.PointIdsList(points=[point_id]),
                wait=True,
            )

    def is_document_registered(self, document_hash: str) -> RegistryEntry | None:
        """Checks if a document is registered in the registry collection.
        
        Args:
            document_hash (str): The hash of the document.
        
        Returns:
            RegistryEntry | None: The document's metadata if registered, None otherwise.
        """
        registry_id = self._generate_point_id(document_hash)
        flt = qm.Filter(
            must=[
                qm.HasIdCondition(has_id=[registry_id])
            ]
        )

        try:
            next_offset = 0
            while True:
                points, next_offset = self._connection.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=flt,
                    limit=1,
                    with_payload=["title", "models"],
                    with_vectors=False,
                    offset=next_offset
                )

                if points:
                    return RegistryEntry(
                        document_hash=document_hash,
                        title=points[0].payload.get("title", ""),
                        models=frozenset(points[0].payload.get("models", []))
                    )
                if next_offset is None:
                    break
        except Exception as e:
            logger.error(f"Error while checking if document is indexed: {e}")
            raise Exception(f"Error while checking if document is indexed: {e}")
        return None

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
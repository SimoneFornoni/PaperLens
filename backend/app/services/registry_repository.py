import logging
import uuid
from typing import List

from backend.app.services.qdrant_connection_provider import QdrantConnectionProvider
from qdrant_client.http import models as qm

logger = logging.getLogger(__name__)

class RegistryRepository:

    def __init__(self, connection: QdrantConnectionProvider):
        self._connection = connection
        self.collection_name = "PaperLens_Registry"
        self._ensure_collection()

    def _ensure_collection(self):
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
        input_str = f"{document_hash}"
        NAMESAPCE_PaperLens = uuid.UUID("12345678-1234-5678-1234-567812345678")
        return str(uuid.uuid5(NAMESAPCE_PaperLens, input_str))

    def register_document(self, document_hash: str, title: str, model: str):
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

    def is_document_registered(self, document_hash: str, model: str) -> bool:
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
                    with_payload=False,
                    with_vectors=False,
                    offset=next_offset
                )

                if points:
                    return True

                if next_offset is None:
                    break
        except Exception as e:
            logger.error(f"Error while checking if document is indexed: {e}")
            raise Exception(f"Error while checking if document is indexed: {e}")
        return False

    def is_title_registered(self, title: str) -> bool:
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

    def list_registered_documents(self, model: str) -> List[str]:
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
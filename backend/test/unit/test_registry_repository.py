from unittest.mock import patch
from types import SimpleNamespace

from app.repositories.registry_repository import RegistryRepository
    
def test_register_document():
    with patch('app.repositories.registry_repository.QdrantConnectionProvider') as MockConnection:
        mock_connection = MockConnection.return_value
        mock_client = mock_connection.client
        mock_client.retrieve.return_value = []

        SUT = RegistryRepository(connection=mock_connection)

        document_hash = "hash1"
        title = "Title 1"
        model = "Model 1"

        SUT.register_document(document_hash, title, model)

        assert mock_client.upsert.call_count == 1
        payload = mock_client.upsert.call_args.kwargs["points"][0].payload
        assert payload == {
            "document_hash": document_hash,
            "title": title,
            "models": [model],
        }

def test_register_document_with_different_model():
    with patch('app.repositories.registry_repository.QdrantConnectionProvider') as MockConnection:
        mock_connection = MockConnection.return_value
        mock_client = mock_connection.client
        mock_client.retrieve.side_effect = [
            [SimpleNamespace(id="point_id", payload={"document_hash": "hash1", "title": "Title 1", "models": ["Model 1"]})],
            []
        ]

        SUT = RegistryRepository(connection=mock_connection)

        document_hash = "hash1"
        title = "Title 1"
        model = "Model 2"

        SUT.register_document(document_hash, title, model)

        assert mock_client.set_payload.call_count == 1
        assert mock_client.set_payload.call_args.kwargs["payload"] == {
            "document_hash": document_hash,
            "title": title,
            "models": ["Model 1", model],
        }


def test_unregister_document_model_preserves_other_models():
    with patch('app.repositories.registry_repository.QdrantConnectionProvider') as MockConnection:
        mock_connection = MockConnection.return_value
        mock_client = mock_connection.client
        mock_client.retrieve.return_value = [
            SimpleNamespace(
                id="point_id",
                payload={"document_hash": "hash1", "title": "Title 1", "models": ["Model 1", "Model 2"]},
            )
        ]

        SUT = RegistryRepository(connection=mock_connection)
        SUT.unregister_document_model("hash1", "Model 2")

        mock_client.set_payload.assert_called_once()
        assert mock_client.set_payload.call_args.kwargs["payload"] == {
            "document_hash": "hash1",
            "title": "Title 1",
            "models": ["Model 1"],
        }
        mock_client.delete.assert_not_called()


def test_unregister_last_document_model_deletes_registry_entry():
    with patch('app.repositories.registry_repository.QdrantConnectionProvider') as MockConnection:
        mock_connection = MockConnection.return_value
        mock_client = mock_connection.client
        mock_client.retrieve.return_value = [
            SimpleNamespace(
                id="point_id",
                payload={"document_hash": "hash1", "title": "Title 1", "models": ["Model 1"]},
            )
        ]

        SUT = RegistryRepository(connection=mock_connection)
        SUT.unregister_document_model("hash1", "Model 1")

        mock_client.delete.assert_called_once()
        assert mock_client.delete.call_args.kwargs["wait"] is True
        mock_client.set_payload.assert_not_called()


def test_is_document_registered():
    with patch('app.repositories.registry_repository.QdrantConnectionProvider') as MockConnection:
        mock_connection = MockConnection.return_value
        mock_client = mock_connection.client

        SUT = RegistryRepository(connection=mock_connection)

        document_hash = "hash1"

        mock_client.scroll.return_value = ([], None)

        result = SUT.is_document_registered(document_hash)

        assert result is None

def test_is_document_registered_with_existing_document():
    with patch('app.repositories.registry_repository.QdrantConnectionProvider') as MockConnection:
        mock_connection = MockConnection.return_value
        mock_client = mock_connection.client

        SUT = RegistryRepository(connection=mock_connection)

        document_hash = "hash1"

        mock_client.scroll.return_value = (
            [SimpleNamespace(id="point_id", payload={"doucument_hash": "hash1", "title": "Title 1", "models": ["Model 1"]})],
            None
        )

        result = SUT.is_document_registered(document_hash)

        assert result is not None
        assert result.title == "Title 1"
        assert "Model 1" in result.models

def test_is_title_registered():
    with patch('app.repositories.registry_repository.QdrantConnectionProvider') as MockConnection:
        mock_connection = MockConnection.return_value
        mock_client = mock_connection.client

        SUT = RegistryRepository(connection=mock_connection)

        title = "Title 1"

        mock_client.scroll.return_value = ([], None)

        result = SUT.is_title_registered(title)

        assert result is False

def test_is_title_registered_with_existing_title():
    with patch('app.repositories.registry_repository.QdrantConnectionProvider') as MockConnection:
        mock_connection = MockConnection.return_value
        mock_client = mock_connection.client

        SUT = RegistryRepository(connection=mock_connection)

        title = "Title 1"

        mock_client.scroll.return_value = ([SimpleNamespace(id="point_id")], None)

        result = SUT.is_title_registered(title)

        assert result is True

def test_list_registered_documents():
    with patch('app.repositories.registry_repository.QdrantConnectionProvider') as MockConnection:
        mock_connection = MockConnection.return_value
        mock_client = mock_connection.client

        SUT = RegistryRepository(connection=mock_connection)

        mock_client.scroll.return_value = (
            [
                SimpleNamespace(id="point_id1", payload={"doucument_hash": "hash1", "title": "Title 1", "models": ["Model 1"]}),
                SimpleNamespace(id="point_id2", payload={"doucument_hash": "hash2", "title": "Title 2", "models": ["Model 1"]})
            ],
            None
        )

        result = SUT.list_registered_documents("Model 1")

        assert len(result) == 2
        assert result[0] == "Title 1"
        assert result[1] == "Title 2"

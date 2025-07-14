import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.utils.vector_store import VectorStore

@pytest.fixture
def mock_settings():
    with patch('app.utils.vector_store.get_settings') as mock:
        mock.return_value = MagicMock(
            VECTOR_STORE_PATH='test/path'
        )
        yield mock

@pytest.fixture
def mock_chroma_client():
    with patch('app.utils.vector_store.chromadb.PersistentClient') as mock:
        client_instance = MagicMock()
        collection = MagicMock()
        collection.count.return_value = 0
        client_instance.get_or_create_collection.return_value = collection
        mock.return_value = client_instance
        yield client_instance

@pytest.fixture
def vector_store(mock_settings, mock_chroma_client):
    return VectorStore()

@pytest.mark.asyncio
async def test_add_documents_success(vector_store):
    # Mock data
    documents = [
        {
            'embedding': [0.1, 0.2, 0.3],
            'content': 'test content',
            'metadata': {'source': 'test'}
        }
    ]
    
    # Execute
    result = await vector_store.add_documents(documents)
    
    # Assert
    assert result is True
    vector_store.collection.add.assert_called_once()
    call_args = vector_store.collection.add.call_args[1]
    assert len(call_args['ids']) == 1
    assert call_args['embeddings'] == [[0.1, 0.2, 0.3]]
    assert call_args['documents'] == ['test content']
    assert call_args['metadatas'] == [{'source': 'test'}]

@pytest.mark.asyncio
async def test_add_documents_invalid_embedding(vector_store):
    # Test data with invalid embedding
    documents = [
        {
            'embedding': 'invalid',  # Should be list of numbers
            'content': 'test content',
            'metadata': {'source': 'test'}
        }
    ]
    
    # Execute
    result = await vector_store.add_documents(documents)
    
    # Assert
    assert result is False
    vector_store.collection.add.assert_not_called()

@pytest.mark.asyncio
async def test_add_documents_error(vector_store):
    # Mock collection to raise error
    vector_store.collection.add.side_effect = Exception('Test error')
    
    # Test data
    documents = [
        {
            'embedding': [0.1, 0.2, 0.3],
            'content': 'test content',
            'metadata': {'source': 'test'}
        }
    ]
    
    # Execute
    result = await vector_store.add_documents(documents)
    
    # Assert
    assert result is False

@pytest.mark.asyncio
async def test_similarity_search_success(vector_store):
    # Mock data
    mock_results = {
        'documents': [['doc1']],
        'metadatas': [[{'source': 'test'}]],
        'distances': [[0.5]]
    }
    vector_store.collection.query.return_value = mock_results
    
    # Execute
    results = await vector_store.similarity_search('test query', k=1)
    
    # Assert
    assert len(results) == 1
    assert results[0]['content'] == 'doc1'
    assert results[0]['metadata'] == {'source': 'test'}
    assert results[0]['distance'] == 0.5
    vector_store.collection.query.assert_called_once()

@pytest.mark.asyncio
async def test_similarity_search_with_filter(vector_store):
    # Mock data
    mock_results = {
        'documents': [[]],
        'metadatas': [[]],
        'distances': [[]]
    }
    vector_store.collection.query.return_value = mock_results
    test_filter = {'source': 'test'}
    
    # Execute
    await vector_store.similarity_search('test query', k=5, filter=test_filter)
    
    # Assert
    vector_store.collection.query.assert_called_once()
    assert vector_store.collection.query.call_args[1]['where'] == test_filter

@pytest.mark.asyncio
async def test_similarity_search_error(vector_store):
    # Mock collection to raise error
    vector_store.collection.query.side_effect = Exception('Test error')
    
    # Execute
    results = await vector_store.similarity_search('test query')
    
    # Assert
    assert results == []

@pytest.mark.asyncio
async def test_delete_documents_success(vector_store):
    # Execute
    result = await vector_store.delete_documents(['1', '2'])
    
    # Assert
    assert result is True
    vector_store.collection.delete.assert_called_once_with(ids=['1', '2'])

@pytest.mark.asyncio
async def test_delete_documents_error(vector_store):
    # Mock collection to raise error
    vector_store.collection.delete.side_effect = Exception('Test error')
    
    # Execute
    result = await vector_store.delete_documents(['1'])
    
    # Assert
    assert result is False

@pytest.mark.asyncio
async def test_update_document_success(vector_store):
    # Test data
    doc_id = 'test_id'
    content = 'updated content'
    embedding = [0.1, 0.2, 0.3]
    metadata = {'source': 'test'}
    
    # Execute
    result = await vector_store.update_document(
        id=doc_id,
        content=content,
        embedding=embedding,
        metadata=metadata
    )
    
    # Assert
    assert result is True
    vector_store.collection.update.assert_called_once_with(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[content],
        metadatas=[metadata]
    )

@pytest.mark.asyncio
async def test_update_document_without_metadata(vector_store):
    # Execute
    result = await vector_store.update_document(
        id='test_id',
        content='content',
        embedding=[0.1],
        metadata=None
    )
    
    # Assert
    assert result is True
    vector_store.collection.update.assert_called_once()
    assert vector_store.collection.update.call_args[1]['metadatas'] is None

@pytest.mark.asyncio
async def test_update_document_error(vector_store):
    # Mock collection to raise error
    vector_store.collection.update.side_effect = Exception('Test error')
    
    # Execute
    result = await vector_store.update_document(
        id='test_id',
        content='content',
        embedding=[0.1]
    )
    
    # Assert
    assert result is False
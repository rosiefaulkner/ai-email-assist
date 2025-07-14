import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from app.utils.embeddings import EmbeddingUtil

@pytest.fixture
def mock_settings():
    with patch('app.utils.embeddings.get_settings') as mock:
        mock.return_value = MagicMock(
            GOOGLE_API_KEY='test_api_key'
        )
        yield mock

@pytest.fixture
def mock_genai():
    with patch('app.utils.embeddings.genai') as mock:
        mock.embed_content = MagicMock()
        yield mock

@pytest.fixture
def embedding_util(mock_settings, mock_genai):
    return EmbeddingUtil()

@pytest.mark.asyncio
async def test_get_embedding_single_text(embedding_util, mock_genai):
    # Mock data
    test_text = 'test text'
    mock_embedding = [0.1] * 768
    mock_genai.embed_content.return_value = {'embedding': mock_embedding}
    
    # Execute
    result = await embedding_util.get_embedding(test_text)
    
    # Assert
    assert result == mock_embedding
    mock_genai.embed_content.assert_called_once_with(
        model='embedding-001',
        content=test_text
    )

@pytest.mark.asyncio
async def test_get_embedding_list_text(embedding_util, mock_genai):
    # Mock data
    test_texts = ['text1', 'text2']
    mock_embedding = [0.1] * 768
    mock_genai.embed_content.return_value = {'embedding': mock_embedding}
    
    # Execute
    result = await embedding_util.get_embedding(test_texts)
    
    # Assert
    assert len(result) == 2
    assert all(r == mock_embedding for r in result)
    assert mock_genai.embed_content.call_count == 2

@pytest.mark.asyncio
async def test_get_embedding_error(embedding_util, mock_genai):
    # Mock error
    mock_genai.embed_content.side_effect = Exception('Test error')
    
    # Execute
    result = await embedding_util.get_embedding('test text')
    
    # Assert
    assert len(result) == 768  # embedding-001 dimension
    assert all(x == 0.0 for x in result)

@pytest.mark.asyncio
async def test_batch_get_embeddings_success(embedding_util, mock_genai):
    # Mock data
    test_texts = ['text1', 'text2', 'text3']
    mock_embedding = [0.1] * 768
    mock_genai.embed_content.return_value = {'embedding': mock_embedding}
    
    # Execute
    result = await embedding_util.batch_get_embeddings(test_texts, batch_size=2)
    
    # Assert
    assert len(result) == 3
    assert all(r == mock_embedding for r in result)
    assert mock_genai.embed_content.call_count == 3

@pytest.mark.asyncio
async def test_batch_get_embeddings_empty_input(embedding_util):
    # Execute
    result = await embedding_util.batch_get_embeddings([])
    
    # Assert
    assert result == []

@pytest.mark.asyncio
async def test_batch_get_embeddings_error(embedding_util, mock_genai):
    # Mock error
    test_texts = ['text1', 'text2']
    mock_genai.embed_content.side_effect = Exception('Test error')
    
    # Execute
    result = await embedding_util.batch_get_embeddings(test_texts)
    
    # Assert
    assert len(result) == 2
    assert all(len(r) == 768 and all(x == 0.0 for x in r) for r in result)

def test_compute_similarity_success(embedding_util):
    # Test data
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    
    # Execute
    result = embedding_util.compute_similarity(vec1, vec2)
    
    # Assert
    assert result == 1.0

def test_compute_similarity_orthogonal(embedding_util):
    # Test data
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    
    # Execute
    result = embedding_util.compute_similarity(vec1, vec2)
    
    # Assert
    assert result == 0.0

def test_compute_similarity_error(embedding_util):
    # Test with invalid input
    with patch('numpy.array', side_effect=Exception('Test error')):
        result = embedding_util.compute_similarity([1.0], [1.0])
        assert result == 0.0

@pytest.mark.asyncio
async def test_get_embedding_retry(embedding_util, mock_genai):
    # Mock genai to fail twice then succeed
    mock_embedding = [0.1] * 768
    mock_genai.embed_content.side_effect = [
        Exception('First failure'),
        Exception('Second failure'),
        {'embedding': mock_embedding}
    ]
    
    # Execute
    result = await embedding_util.get_embedding('test text')
    
    # Assert
    assert result == mock_embedding
    assert mock_genai.embed_content.call_count == 3

@pytest.mark.asyncio
async def test_batch_get_embeddings_retry(embedding_util, mock_genai):
    # Mock genai to fail twice then succeed
    mock_embedding = [0.1] * 768
    mock_genai.embed_content.side_effect = [
        Exception('First failure'),
        Exception('Second failure'),
        {'embedding': mock_embedding}
    ]
    
    # Execute
    result = await embedding_util.batch_get_embeddings(['test text'])
    
    # Assert
    assert result == [mock_embedding]
    assert mock_genai.embed_content.call_count == 3
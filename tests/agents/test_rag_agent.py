import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.agents.rag_agent import RAGAgent

@pytest.fixture
def mock_settings():
    with patch('app.agents.rag_agent.get_settings') as mock:
        mock.return_value = MagicMock(
            MAX_DOCUMENTS=5,
            SIMILARITY_THRESHOLD=0.7
        )
        yield mock

@pytest.fixture
def mock_vector_store():
    with patch('app.agents.rag_agent.VectorStore') as mock:
        instance = mock.return_value
        instance.similarity_search = AsyncMock()
        instance.add_documents = AsyncMock()
        yield instance

@pytest.fixture
def mock_embedding_util():
    with patch('app.agents.rag_agent.EmbeddingUtil') as mock:
        instance = mock.return_value
        instance.get_embedding = AsyncMock()
        yield instance

@pytest.fixture
def mock_llm():
    with patch('app.agents.rag_agent.GeminiAgent') as mock:
        instance = mock.return_value
        instance.generate_response = AsyncMock()
        instance.analyze_relevance = AsyncMock()
        yield instance

@pytest.fixture
def rag_agent(mock_settings, mock_vector_store, mock_embedding_util, mock_llm):
    return RAGAgent()

@pytest.mark.asyncio
async def test_process_query_success(rag_agent):
    # Mock data
    mock_docs = [
        {
            'content': 'test content 1',
            'metadata': {'source': 'test1'},
            'score': 0.8
        }
    ]
    mock_response = {
        'answer': 'test answer',
        'metadata': {'model': 'test-model'}
    }
    
    # Setup mocks
    rag_agent.retrieve_relevant_documents = AsyncMock(return_value=mock_docs)
    rag_agent.llm.generate_response = AsyncMock(return_value=mock_response)
    
    # Execute
    result = await rag_agent.process_query('test query')
    
    # Assert
    assert result['answer'] == 'test answer'
    assert len(result['sources']) == 1
    assert result['sources'][0]['content'] == 'test content 1'
    assert result['sources'][0]['relevance_score'] == 0.8

@pytest.mark.asyncio\async def test_process_query_no_relevant_docs(rag_agent):
    # Setup mocks
    rag_agent.retrieve_relevant_documents = AsyncMock(return_value=[])
    rag_agent.llm.generate_response = AsyncMock(return_value={'answer': None})
    
    # Execute
    result = await rag_agent.process_query('test query')
    
    # Assert
    assert "couldn't find any relevant information" in result['answer']
    assert result['sources'] == []

@pytest.mark.asyncio
async def test_process_query_error(rag_agent):
    # Setup mock to raise exception
    rag_agent.retrieve_relevant_documents = AsyncMock(side_effect=Exception('Test error'))
    
    # Execute
    result = await rag_agent.process_query('test query')
    
    # Assert
    assert result['error'] == 'Test error'
    assert result['answer'] is None
    assert result['sources'] == []
    assert result['metadata']['error_type'] == 'Exception'

@pytest.mark.asyncio
async def test_retrieve_relevant_documents(rag_agent):
    # Mock data
    mock_candidates = [
        {'content': 'doc1', 'metadata': {'source': 'test1'}},
        {'content': 'doc2', 'metadata': {'source': 'test2'}}
    ]
    
    # Setup mocks
    rag_agent.vector_store.similarity_search = AsyncMock(return_value=mock_candidates)
    rag_agent.llm.analyze_relevance = AsyncMock(side_effect=[0.8, 0.6])
    
    # Execute
    result = await rag_agent.retrieve_relevant_documents('test query')
    
    # Assert
    assert len(result) == 1  # Only doc1 should be included (score >= 0.7)
    assert result[0]['content'] == 'doc1'
    assert result[0]['score'] == 0.8

def test_prepare_context(rag_agent):
    # Test data
    documents = [
        {
            'content': 'test content 1',
            'metadata': {'source': 'test1'},
            'score': 0.8
        },
        {
            'content': 'test content 2',
            'metadata': {'source': 'test2'},
            'score': 0.7
        }
    ]
    
    # Execute
    result = rag_agent._prepare_context(documents)
    
    # Assert
    assert len(result) == 2
    assert 'test content 1' in result[0]
    assert 'Source: test1' in result[0]
    assert 'test content 2' in result[1]
    assert 'Source: test2' in result[1]

@pytest.mark.asyncio
async def test_add_document_success(rag_agent):
    # Mock data
    content = 'test content'
    metadata = {'source': 'test'}
    mock_embedding = [0.1, 0.2, 0.3]
    
    # Setup mocks
    rag_agent.embedding_util.get_embedding = AsyncMock(return_value=mock_embedding)
    rag_agent.vector_store.add_documents = AsyncMock()
    
    # Execute
    result = await rag_agent.add_document(content, metadata)
    
    # Assert
    assert result is True
    rag_agent.embedding_util.get_embedding.assert_called_once_with(content)
    rag_agent.vector_store.add_documents.assert_called_once()

@pytest.mark.asyncio
async def test_add_document_failure(rag_agent):
    # Setup mock to raise exception
    rag_agent.embedding_util.get_embedding = AsyncMock(side_effect=Exception('Test error'))
    
    # Execute
    result = await rag_agent.add_document('test content')
    
    # Assert
    assert result is False
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.graph.workflow import RAGWorkflow

@pytest.fixture
def mock_rag_agent():
    with patch('app.graph.workflow.RAGAgent') as mock:
        instance = mock.return_value
        instance.retrieve_relevant_documents = AsyncMock()
        instance.process_query = AsyncMock()
        yield instance

@pytest.fixture
def mock_llm_agent():
    with patch('app.graph.workflow.GeminiAgent') as mock:
        instance = mock.return_value
        instance.generate_response = AsyncMock()
        yield instance

@pytest.fixture
def mock_state_graph():
    with patch('app.graph.workflow.StateGraph') as mock:
        instance = mock.return_value
        instance.add_node = MagicMock()
        instance.add_edge = MagicMock()
        instance.add_conditional_edges = MagicMock()
        instance.set_entry_point = MagicMock()
        instance.compile = MagicMock()
        yield instance

@pytest.fixture
def workflow(mock_rag_agent, mock_llm_agent, mock_state_graph):
    return RAGWorkflow()

def test_build_graph(workflow, mock_state_graph):
    workflow._build_graph()
    
    # Assert nodes were added
    assert mock_state_graph.return_value.add_node.call_count == 3
    assert mock_state_graph.return_value.add_edge.call_count >= 2
    mock_state_graph.return_value.add_conditional_edges.assert_called_once()
    mock_state_graph.return_value.set_entry_point.assert_called_once_with('retrieve')
    mock_state_graph.return_value.compile.assert_called_once()

@pytest.mark.asyncio
async def test_retrieve_context(workflow):
    # Mock data
    state = {'query': 'test query'}
    mock_docs = [{'content': 'test content', 'metadata': {'source': 'test'}}]
    workflow.rag_agent.retrieve_relevant_documents.return_value = mock_docs
    
    # Execute
    result = await workflow._retrieve_context(state)
    
    # Assert
    assert result['query'] == 'test query'
    assert result['context'] == mock_docs
    assert result['attempt'] == 0
    workflow.rag_agent.retrieve_relevant_documents.assert_called_once_with('test query')

@pytest.mark.asyncio
async def test_generate_response(workflow):
    # Mock data
    state = {
        'query': 'test query',
        'context': [{'content': 'test content'}],
        'attempt': 0
    }
    mock_response = {
        'answer': 'test answer',
        'sources': [],
        'metadata': {}
    }
    workflow.rag_agent.process_query.return_value = mock_response
    
    # Execute
    result = await workflow._generate_response(state)
    
    # Assert
    assert result['response'] == mock_response
    assert result['attempt'] == 1
    workflow.rag_agent.process_query.assert_called_once_with(
        query='test query',
        context=['test content']
    )

@pytest.mark.asyncio
async def test_validate_response_success(workflow):
    # Mock data
    state = {
        'response': {
            'answer': 'test answer',
            'metadata': {}
        }
    }
    workflow.llm_agent.generate_response.return_value = {'answer': '0.8'}
    
    # Execute
    result = await workflow._validate_response(state)
    
    # Assert
    assert result['valid'] is True
    assert result['quality_score'] == 0.8

@pytest.mark.asyncio
async def test_validate_response_error(workflow):
    # Mock data
    state = {
        'response': {
            'error': 'test error',
            'answer': None
        }
    }
    
    # Execute
    result = await workflow._validate_response(state)
    
    # Assert
    assert result['valid'] is False
    assert result['error'] == 'test error'

def test_should_regenerate(workflow):
    # Test cases
    assert workflow._should_regenerate({'valid': False, 'attempt': 1}) is True
    assert workflow._should_regenerate({'valid': True, 'attempt': 1}) is False
    assert workflow._should_regenerate({'valid': False, 'attempt': 3}) is False

@pytest.mark.asyncio
async def test_run_success(workflow):
    # Mock data
    inputs = {'query': 'test query'}
    mock_result = {
        'response': {
            'answer': 'test answer',
            'sources': [],
            'metadata': {}
        },
        'quality_score': 0.8,
        'attempt': 1
    }
    workflow.graph.arun = AsyncMock(return_value=mock_result)
    
    # Execute
    result = await workflow.run(inputs)
    
    # Assert
    assert result['answer'] == 'test answer'
    assert 'sources' in result
    assert result['metadata']['quality_score'] == 0.8
    assert result['metadata']['attempts'] == 1

@pytest.mark.asyncio
async def test_run_error(workflow):
    # Mock data
    inputs = {'query': 'test query'}
    workflow.graph.arun = AsyncMock(side_effect=Exception('Test error'))
    
    # Execute
    result = await workflow.run(inputs)
    
    # Assert
    assert result['error'] == 'Test error'
    assert result['answer'] is None
    assert result['sources'] == []
    assert result['metadata']['error_type'] == 'Exception'
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.agents.gemini_agent import GeminiAgent

@pytest.fixture
def mock_settings():
    with patch('app.agents.gemini_agent.get_settings') as mock:
        mock.return_value = MagicMock(
            GOOGLE_API_KEY='test_api_key',
            GEMINI_MODEL='test-model',
            TEMPERATURE=0.7,
            TOP_P=0.8,
            TOP_K=40,
            MAX_TOKENS=1000
        )
        yield mock

@pytest.fixture
def gemini_agent(mock_settings):
    with patch('google.generativeai.configure'), \
         patch('google.generativeai.GenerativeModel'):
        agent = GeminiAgent()
        yield agent

@pytest.mark.asyncio
async def test_generate_response_empty_query(gemini_agent):
    result = await gemini_agent.generate_response('')
    assert result['answer'] is None
    assert result['error'] == 'Query cannot be empty'
    assert result['metadata']['error_type'] == 'ValueError'

@pytest.mark.asyncio
async def test_generate_response_success(gemini_agent):
    mock_response = MagicMock(
        text='Test response',
        finish_reason='stop',
        prompt_token_count=10,
        completion_token_count=20
    )
    gemini_agent.model.generate_content_async = AsyncMock(return_value=mock_response)
    
    result = await gemini_agent.generate_response('test query')
    
    assert result['answer'] == 'Test response'
    assert result['metadata']['model'] == 'gemini-pro'
    assert result['metadata']['finish_reason'] == 'stop'
    assert result['metadata']['prompt_tokens'] == 10
    assert result['metadata']['completion_tokens'] == 20

@pytest.mark.asyncio
async def test_generate_response_with_context(gemini_agent):
    mock_response = MagicMock(text='Test response with context')
    gemini_agent.model.generate_content_async = AsyncMock(return_value=mock_response)
    
    context = ['context1', 'context2']
    result = await gemini_agent.generate_response('test query', context=context)
    
    assert result['answer'] == 'Test response with context'

@pytest.mark.asyncio
async def test_generate_response_model_error(gemini_agent):
    gemini_agent.model.generate_content_async = AsyncMock(return_value=MagicMock(text=''))
    
    result = await gemini_agent.generate_response('test query')
    
    assert result['answer'] is None
    assert result['error'] == 'No response generated from the model'
    assert result['metadata']['error_type'] == 'ModelError'

@pytest.mark.asyncio
async def test_generate_response_exception(gemini_agent):
    gemini_agent.model.generate_content_async = AsyncMock(side_effect=Exception('Test error'))
    
    result = await gemini_agent.generate_response('test query')
    
    assert result['answer'] is None
    assert result['error'] == 'Test error'
    assert result['metadata']['error_type'] == 'Exception'

def test_prepare_prompt_without_context(gemini_agent):
    query = 'test query'
    result = gemini_agent._prepare_prompt(query)
    assert result == query

def test_prepare_prompt_with_context(gemini_agent):
    query = 'test query'
    context = ['context1', 'context2']
    result = gemini_agent._prepare_prompt(query, context)
    assert 'context1' in result
    assert 'context2' in result
    assert query in result

@pytest.mark.asyncio
async def test_analyze_relevance_success(gemini_agent):
    mock_response = MagicMock(text='0.8')
    gemini_agent.model.generate_content_async = AsyncMock(return_value=mock_response)
    
    score = await gemini_agent.analyze_relevance('test query', 'test document')
    assert score == 0.8

@pytest.mark.asyncio
async def test_analyze_relevance_exception(gemini_agent):
    gemini_agent.model.generate_content_async = AsyncMock(side_effect=Exception('Test error'))
    
    score = await gemini_agent.analyze_relevance('test query', 'test document')
    assert score == 0.0
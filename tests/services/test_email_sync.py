import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.email_sync import EmailSyncService

@pytest.fixture
def mock_settings():
    with patch('app.services.email_sync.get_settings') as mock:
        mock.return_value = MagicMock()
        yield mock

@pytest.fixture
def mock_gmail_client():
    with patch('app.services.email_sync.GmailClient') as mock:
        instance = mock.return_value
        instance.get_messages = AsyncMock()
        yield instance

@pytest.fixture
def mock_vector_store():
    with patch('app.services.email_sync.VectorStore') as mock:
        instance = mock.return_value
        instance.add_documents = AsyncMock()
        yield instance

@pytest.fixture
def mock_embedding_util():
    with patch('app.services.email_sync.EmbeddingUtil') as mock:
        instance = mock.return_value
        instance.batch_get_embeddings = AsyncMock()
        yield instance

@pytest.fixture
def email_sync_service(mock_settings, mock_gmail_client, mock_vector_store, mock_embedding_util):
    return EmailSyncService()

@pytest.mark.asyncio
async def test_sync_emails_success(email_sync_service, mock_gmail_client, mock_embedding_util, mock_vector_store):
    # Mock data
    mock_messages = [
        {
            'id': '1',
            'snippet': 'test email 1',
            'internalDate': '2024-01-01',
            'subject': 'Test Subject 1'
        },
        {
            'id': '2',
            'snippet': 'test email 2',
            'internalDate': '2024-01-02',
            'subject': 'Test Subject 2'
        }
    ]
    mock_embeddings = [[0.1, 0.2], [0.3, 0.4]]
    
    # Setup mocks
    mock_gmail_client.get_messages.return_value = mock_messages
    mock_embedding_util.batch_get_embeddings.return_value = mock_embeddings
    mock_vector_store.add_documents.return_value = True
    
    # Execute
    result = await email_sync_service.sync_emails(max_results=2)
    
    # Assert
    assert result is True
    mock_gmail_client.get_messages.assert_called_once_with(max_results=2)
    mock_embedding_util.batch_get_embeddings.assert_called_once()
    mock_vector_store.add_documents.assert_called_once()

@pytest.mark.asyncio
async def test_sync_emails_no_messages(email_sync_service, mock_gmail_client):
    # Setup mock
    mock_gmail_client.get_messages.return_value = []
    
    # Execute
    result = await email_sync_service.sync_emails()
    
    # Assert
    assert result is True
    mock_gmail_client.get_messages.assert_called_once()

@pytest.mark.asyncio
async def test_sync_emails_empty_snippets(email_sync_service, mock_gmail_client, mock_embedding_util):
    # Mock data with empty snippets
    mock_messages = [
        {'id': '1', 'snippet': '', 'internalDate': '2024-01-01'},
        {'id': '2', 'snippet': '', 'internalDate': '2024-01-02'}
    ]
    
    # Setup mock
    mock_gmail_client.get_messages.return_value = mock_messages
    
    # Execute
    result = await email_sync_service.sync_emails()
    
    # Assert
    assert result is True
    mock_embedding_util.batch_get_embeddings.assert_not_called()

@pytest.mark.asyncio
async def test_sync_emails_embedding_failure(email_sync_service, mock_gmail_client, mock_embedding_util):
    # Mock data
    mock_messages = [{'id': '1', 'snippet': 'test', 'internalDate': '2024-01-01'}]
    
    # Setup mocks
    mock_gmail_client.get_messages.return_value = mock_messages
    mock_embedding_util.batch_get_embeddings.return_value = [None]
    
    # Execute
    result = await email_sync_service.sync_emails()
    
    # Assert
    assert result is False

@pytest.mark.asyncio
async def test_sync_emails_vector_store_failure(email_sync_service, mock_gmail_client, mock_embedding_util, mock_vector_store):
    # Mock data
    mock_messages = [{'id': '1', 'snippet': 'test', 'internalDate': '2024-01-01'}]
    mock_embeddings = [[0.1, 0.2]]
    
    # Setup mocks
    mock_gmail_client.get_messages.return_value = mock_messages
    mock_embedding_util.batch_get_embeddings.return_value = mock_embeddings
    mock_vector_store.add_documents.return_value = False
    
    # Execute
    result = await email_sync_service.sync_emails()
    
    # Assert
    assert result is False

@pytest.mark.asyncio
async def test_sync_emails_exception(email_sync_service, mock_gmail_client):
    # Setup mock to raise exception
    mock_gmail_client.get_messages.side_effect = Exception('Test error')
    
    # Execute
    result = await email_sync_service.sync_emails()
    
    # Assert
    assert result is False

@pytest.mark.asyncio
async def test_start_service(email_sync_service):
    # Mock sync_emails and sleep to test the service loop
    email_sync_service.sync_emails = AsyncMock()
    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
        mock_sleep.side_effect = [None, Exception('Stop loop')]  # Run once then stop
        
        # Execute
        with pytest.raises(Exception, match='Stop loop'):
            await email_sync_service.start()
        
        # Assert
        assert email_sync_service.sync_emails.called
        assert mock_sleep.called

def test_stop_service(email_sync_service):
    # Execute
    email_sync_service.stop()
    # Currently just testing that it doesn't raise any exceptions
    # Add more assertions if stop() implementation is enhanced
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.tools.gmail import GmailClient

@pytest.fixture
def mock_credentials():
    with patch('app.tools.gmail.pickle.load') as mock_load, \
         patch('app.tools.gmail.pickle.dump') as mock_dump:
        creds = MagicMock()
        creds.valid = True
        mock_load.return_value = creds
        yield creds

@pytest.fixture
def mock_gmail_service():
    with patch('app.tools.gmail.build') as mock_build:
        service = MagicMock()
        service._http = MagicMock()
        service._http.credentials = MagicMock()
        service._http.credentials.apply = MagicMock()
        mock_build.return_value = service
        yield service

@pytest.fixture
def mock_aiohttp_session():
    with patch('aiohttp.ClientSession') as mock:
        session = AsyncMock()
        context_manager = AsyncMock()
        context_manager.__aenter__.return_value = session
        mock.return_value = context_manager
        yield session

@pytest.fixture
def gmail_client(mock_credentials, mock_gmail_service):
    with patch('os.path.exists', return_value=True):
        client = GmailClient()
        yield client

def test_init_with_valid_token(mock_credentials, mock_gmail_service):
    with patch('os.path.exists', return_value=True):
        client = GmailClient()
        assert client.service is not None

def test_init_with_expired_token(mock_credentials, mock_gmail_service):
    mock_credentials.valid = False
    mock_credentials.expired = True
    mock_credentials.refresh_token = True
    
    with patch('os.path.exists', return_value=True):
        client = GmailClient()
        mock_credentials.refresh.assert_called_once()

def test_init_without_token(mock_gmail_service):
    with patch('os.path.exists', return_value=False), \
         patch('google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file') as mock_flow:
        flow = MagicMock()
        flow.run_local_server.return_value = MagicMock(valid=True)
        mock_flow.return_value = flow
        
        client = GmailClient()
        mock_flow.assert_called_once()
        flow.run_local_server.assert_called_once()

@pytest.mark.asyncio
async def test_get_messages_success(gmail_client, mock_aiohttp_session):
    # Mock list response
    list_response = MagicMock(status=200)
    list_response.json = AsyncMock(return_value={
        'messages': [
            {'id': '1'},
            {'id': '2'}
        ]
    })
    
    # Mock message detail response
    detail_response = MagicMock(status=200)
    detail_response.json = AsyncMock(return_value={
        'id': '1',
        'snippet': 'Test snippet',
        'payload': {
            'headers': [
                {'name': 'Subject', 'value': 'Test Subject'},
                {'name': 'From', 'value': 'test@example.com'}
            ]
        }
    })
    
    mock_aiohttp_session.get = AsyncMock(side_effect=[list_response, detail_response, detail_response])
    
    messages = await gmail_client.get_messages(max_results=2)
    
    assert len(messages) == 2
    assert messages[0]['id'] == '1'
    assert messages[0]['snippet'] == 'Test snippet'
    assert messages[0]['subject'] == 'Test Subject'
    assert messages[0]['from'] == 'test@example.com'

@pytest.mark.asyncio
async def test_get_messages_list_error(gmail_client, mock_aiohttp_session):
    error_response = MagicMock(status=400)
    mock_aiohttp_session.get = AsyncMock(return_value=error_response)
    
    messages = await gmail_client.get_messages()
    assert messages == []

@pytest.mark.asyncio
async def test_get_messages_detail_error(gmail_client, mock_aiohttp_session):
    # Mock list response success
    list_response = MagicMock(status=200)
    list_response.json = AsyncMock(return_value={'messages': [{'id': '1'}]})
    
    # Mock detail response error
    detail_response = MagicMock(status=400)
    
    mock_aiohttp_session.get = AsyncMock(side_effect=[list_response, detail_response])
    
    messages = await gmail_client.get_messages()
    assert messages == []

@pytest.mark.asyncio
async def test_get_last_email_success(gmail_client):
    mock_message = {
        'id': '1',
        'snippet': 'Test snippet',
        'subject': 'Test Subject',
        'from': 'test@example.com'
    }
    gmail_client._fetch_messages = AsyncMock(return_value=[mock_message])
    
    last_email = await gmail_client.get_last_email()
    assert last_email == mock_message

@pytest.mark.asyncio
async def test_get_last_email_no_messages(gmail_client):
    gmail_client._fetch_messages = AsyncMock(return_value=[])
    
    last_email = await gmail_client.get_last_email()
    assert last_email is None

@pytest.mark.asyncio
async def test_get_message_by_id_success(gmail_client, mock_aiohttp_session):
    response = MagicMock(status=200)
    response.json = AsyncMock(return_value={
        'id': '1',
        'snippet': 'Test snippet'
    })
    mock_aiohttp_session.get = AsyncMock(return_value=response)
    
    message = await gmail_client.get_message_by_id('1')
    assert message['id'] == '1'
    assert message['snippet'] == 'Test snippet'

@pytest.mark.asyncio
async def test_get_message_by_id_error(gmail_client, mock_aiohttp_session):
    mock_aiohttp_session.get.side_effect = Exception('Test error')
    
    message = await gmail_client.get_message_by_id('1')
    assert message is None
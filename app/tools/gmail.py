import os.path
import pickle
from typing import Dict, List, Optional

import aiohttp
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import HttpRequest


class GmailClient:
    """Gmail client for accessing and managing emails.

    This class handles Gmail API authentication and provides methods
    for reading emails from the authenticated user's inbox.
    """

    SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

    def __init__(self):
        self.service = None
        self._authenticate()

    def _authenticate(self) -> None:
        """Authenticate with Gmail API using OAuth2.

        Handles token management and OAuth2 flow for Gmail API access.
        """
        creds = None
        if os.path.exists("token.pickle"):
            with open("token.pickle", "rb") as token:
                creds = pickle.load(token)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json", self.SCOPES
                )
                creds = flow.run_local_server(port=0)

            with open("token.pickle", "wb") as token:
                pickle.dump(creds, token)

        self.service = build("gmail", "v1", credentials=creds)

    async def get_messages(self, max_results: int = 10) -> List[Dict[str, str]]:
        """Retrieve messages from the user's inbox.

        Args:
            max_results: Maximum number of messages to retrieve

        Returns:
            List of message dictionaries containing id and snippet
        """
        try:
            request = self.service.users().messages().list(userId="me", labelIds=["INBOX"], maxResults=max_results)
            async with aiohttp.ClientSession() as session:
                async with session.get(request.uri, headers=request.headers) as response:
                    results = await response.json()

            messages = results.get("messages", [])
            if not messages:
                return []

            message_list = []
            for message in messages:
                request = self.service.users().messages().get(userId="me", id=message["id"])
                async with aiohttp.ClientSession() as session:
                    async with session.get(request.uri, headers=request.headers) as response:
                        msg = await response.json()
                message_list.append({"id": message["id"], "snippet": msg["snippet"]})

            return message_list

        except Exception as e:
            print(f"Error retrieving messages: {str(e)}")
            return []

    async def get_message_by_id(self, message_id: str) -> Optional[Dict[str, str]]:
        """Retrieve a specific message by its ID.

        Args:
            message_id: The ID of the message to retrieve

        Returns:
            Message dictionary containing id and snippet, or None if not found
        """
        try:
            request = self.service.users().messages().get(userId="me", id=message_id)
            async with aiohttp.ClientSession() as session:
                async with session.get(request.uri, headers=request.headers) as response:
                    msg = await response.json()

            return {"id": message_id, "snippet": msg["snippet"]}

        except Exception as e:
            print(f"Error retrieving message {message_id}: {str(e)}")
            return None

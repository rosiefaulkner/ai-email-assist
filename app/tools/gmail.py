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
        return await self._fetch_messages(max_results)

    async def get_last_email(self) -> Optional[Dict[str, str]]:
        """Retrieve only the last email from the inbox.

        Returns:
            Dictionary containing id and snippet of the last email, or None if no emails found
        """
        messages = await self._fetch_messages(max_results=1)
        print(f"Last email details: {messages[0]}")
        return messages[0] if messages else None

    async def _fetch_messages(self, max_results: int = 10) -> List[Dict[str, str]]:
        """Internal method to fetch messages from Gmail API.

        Args:
            max_results: Maximum number of messages to retrieve

        Returns:
            List of message dictionaries containing id and snippet
        """
        try:
            request = (
                self.service.users()
                .messages()
                .list(userId="me", labelIds=["INBOX"], maxResults=max_results)
            )
            
            credentials = self.service._http.credentials
            headers = request.headers.copy()
            credentials.apply(headers)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    request.uri, headers=headers
                ) as response:
                    if response.status != 200:
                        print(f"Error fetching message list: {response.status}")
                        return []
                    results = await response.json()

            messages = results.get("messages", [])
            if not messages:
                return []

            message_list = []
            for message in messages:
                detail_request = self.service.users().messages().get(userId="me", id=message["id"])
                detail_headers = detail_request.headers.copy()
                credentials.apply(detail_headers)
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        detail_request.uri, headers=detail_headers
                    ) as response:
                        if response.status == 200:
                            msg_details = await response.json()
                            message_list.append({
                                "id": message["id"],
                                "snippet": msg_details.get("snippet", "No preview available"),
                                "subject": next((header["value"] for header in msg_details.get("payload", {}).get("headers", []) if header["name"].lower() == "subject"), "No subject"),
                                "from": next((header["value"] for header in msg_details.get("payload", {}).get("headers", []) if header["name"].lower() == "from"), "Unknown sender"),
                                "date": msg_details.get("internalDate", "Unknown date"),
                                "attachments": msg_details.get("attachments", []),
                                "threadId": msg_details.get("threadId", "No threadId available"),
                                "raw": msg_details.get("raw", "No raw available"),
                            })
                        else:
                            print(f"Error fetching message {message['id']}: {response.status}")

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
                async with session.get(
                    request.uri, headers=request.headers
                ) as response:
                    msg = await response.json()

            return {"id": message_id, "snippet": msg["snippet"]}

        except Exception as e:
            print(f"Error retrieving message {message_id}: {str(e)}")
            return None
# import asyncio

# async def main():
#     client = GmailClient()
#     last_email = await client.get_last_email()
#     print("Last email details:")
#     print(f"From: {last_email['from']}")
#     print(f"Subject: {last_email['subject']}")
#     print(f"Snippet: {last_email['snippet']}")

# if __name__ == "__main__":
#     asyncio.run(main())
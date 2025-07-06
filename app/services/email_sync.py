import asyncio
from typing import Dict, List, Any

from ..tools.gmail import GmailClient
from ..utils.embeddings import EmbeddingUtil
from ..utils.vector_store import VectorStore
from ..config import get_settings

class EmailSyncService:
    def __init__(self):
        self.settings = get_settings()
        self.gmail_client = GmailClient()
        self.vector_store = VectorStore()
        self.embedding_util = EmbeddingUtil()
        self.sync_interval = 300  # 5 minutes

    async def start(self):
        """Start the email sync service."""
        while True:
            try:
                await self.sync_emails()
                await asyncio.sleep(self.sync_interval)
            except Exception as e:
                print(f"Error in email sync: {str(e)}")
                await asyncio.sleep(60)  # Wait a minute before retrying

    async def sync_emails(self, max_results: int = 50) -> bool:
        """Sync emails from Gmail to vector store.

        Args:
            max_results: Maximum number of emails to sync

        Returns:
            bool: True if sync was successful
        """
        try:
            # Get emails from Gmail
            messages = await self.gmail_client.get_messages(max_results=max_results)
            if not messages:
                return True

            # Process each email
            for message in messages:
                # Generate embedding for email content
                embedding = await self.embedding_util.get_embedding(message['snippet'])

                # Prepare metadata
                metadata = {
                    'email_id': message['id'],
                    'source': 'gmail',
                    'type': 'email'
                }

                # Add to vector store
                await self.vector_store.add_documents([
                    {
                        'content': message['snippet'],
                        'embedding': embedding,
                        'metadata': metadata
                    }
                ])

            return True

        except Exception as e:
            print(f"Error syncing emails: {str(e)}")
            return False

    def stop(self):
        """Stop the email sync service."""
        # Cleanup resources if needed
        pass
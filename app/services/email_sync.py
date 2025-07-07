import asyncio
from typing import Any, Dict, List

from ..config import get_settings
from ..tools.gmail import GmailClient
from ..utils.embeddings import EmbeddingUtil
from ..utils.vector_store import VectorStore


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
            print("Starting email sync...")
            # Get emails from Gmail
            messages = await self.gmail_client.get_messages(max_results=max_results)
            if not messages:
                print("No new messages to sync")
                return True

            print(f"Found {len(messages)} messages to sync")
            successful_syncs = 0
            failed_syncs = 0

            # Process each email
            for message in messages:
                try:
                    if not message.get("snippet"):
                        print(f"Skipping message {message.get('id')}: No content")
                        continue

                    # Generate embedding for email content
                    embedding = await self.embedding_util.get_embedding(
                        message["snippet"]
                    )
                    if embedding is None:
                        print(
                            f"Failed to generate embedding for message {message.get('id')}"
                        )
                        failed_syncs += 1
                        continue

                    # Prepare metadata
                    metadata = {
                        "email_id": message["id"],
                        "source": "gmail",
                        "type": "email",
                        "timestamp": message.get("internalDate"),
                        "subject": message.get("subject", "No subject"),
                    }

                    # Add to vector store
                    await self.vector_store.add_documents(
                        [
                            {
                                "content": message["snippet"],
                                "embedding": embedding,
                                "metadata": metadata,
                            }
                        ]
                    )
                    successful_syncs += 1

                except Exception as e:
                    print(f"Error processing message {message.get('id')}: {str(e)}")
                    failed_syncs += 1

            print(
                f"Sync completed: {successful_syncs} successful, {failed_syncs} failed"
            )
            return successful_syncs > 0 or failed_syncs == 0

        except Exception as e:
            print(f"Error in sync_emails: {str(e)}")
            return False

    def stop(self):
        """Stop the email sync service."""
        # Cleanup resources if needed
        pass

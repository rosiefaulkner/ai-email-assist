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
        """
        Synchronize emails from Gmail and store them in the vector store.
        :param max_results: Maximum number of emails to sync.
        :return: True if sync was successful, False otherwise.
        """
        try:
            print("Starting email sync...")
            messages = await self.gmail_client.get_messages(max_results=max_results)
            if not messages:
                print("No new messages to sync")
                return True

            print(f"Found {len(messages)} messages to sync")
            successful_syncs = 0
            failed_syncs = 0

            # Process emails in batches for more efficient embedding
            batch_size = 10
            for i in range(0, len(messages), batch_size):
                batch = messages[i:i + batch_size]
                snippets = [msg.get("snippet", "") for msg in batch]
                
                # Skip empty snippets
                valid_indices = [i for i, s in enumerate(snippets) if s]
                if not valid_indices:
                    continue
                    
                valid_snippets = [snippets[i] for i in valid_indices]
                valid_messages = [batch[i] for i in valid_indices]

                # Generate embeddings for batch
                embeddings = await self.embedding_util.batch_get_embeddings(valid_snippets)
                
                # Prepare documents for vector store
                documents = []
                for msg, embedding in zip(valid_messages, embeddings):
                    if embedding is None:
                        failed_syncs += 1
                        continue
                        
                    metadata = {
                        "email_id": str(msg["id"]),  # Ensure string type
                        "source": "gmail",
                        "type": "email",
                        "timestamp": str(msg.get("internalDate", "")),  # Ensure string type
                        "subject": str(msg.get("subject", "No subject"))  # Ensure string type
                    }
                    
                    documents.append({
                        "content": msg["snippet"],
                        "embedding": embedding,
                        "metadata": metadata
                    })
                
                if documents:
                    success = await self.vector_store.add_documents(documents)
                    if success:
                        successful_syncs += len(documents)
                    else:
                        failed_syncs += len(documents)
            print(f"Documents: {documents}")
            print(f"Sync completed: {successful_syncs} successful, {failed_syncs} failed")
            return documents

        except Exception as e:
            print(f"Error in sync_emails: {str(e)}")
            return False

    def stop(self):
        """Stop the email sync service."""
        # Cleanup resources if needed
        pass

if __name__ == "__main__":
    import logging
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create and run the service
    service = EmailSyncService()
    try:
        logging.info("Starting email sync service...")
        asyncio.run(service.sync_emails())
        logging.info("Email sync completed")
    except Exception as e:
        logging.error(f"Error running email sync service: {str(e)}")
        raise

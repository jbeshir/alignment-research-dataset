from typing import List
from pydantic import BaseModel
from align_data.embeddings.embedding_utils import Embedding


class MySQLVectorEntry(BaseModel):
    """
    Model for MySQL vector entries that mirrors PineconeEntry functionality.
    
    This model represents an article with its embeddings that will be stored
    in the MySQL article_embeddings table.
    """
    article_hash_id: str
    embeddings: List[Embedding]

    def __repr__(self):
        def make_small(chunk: str) -> str:
            return (chunk[:45] + " [...] " + chunk[-45:]) if len(chunk) > 100 else chunk

        def display_chunks(chunks_lst: List[Embedding]) -> str:
            chunks = ", ".join(f'"{make_small(chunk.text)}"' for chunk in chunks_lst)
            return (
                f"[{chunks[:450]} [...] {chunks[-450:]} ]"
                if len(chunks) > 1000
                else f"[{chunks}]"
            )

        return f"MySQLVectorEntry(article_hash_id={self.article_hash_id!r}, text_chunks={display_chunks(self.embeddings)})"

    @property
    def chunk_num(self) -> int:
        """Return the number of embedding chunks for this article."""
        return len(self.embeddings)

    def create_mysql_records(self) -> List[dict]:
        """
        Convert embeddings to MySQL record format for insertion into article_embeddings table.
        
        Returns:
            List[dict]: List of dictionaries ready for MySQL insertion, each containing:
                - article_hash_id: The article's hash ID
                - chunk_id: Unique identifier for the chunk (article_hash_id + text hash)
                - embedding_vector: The embedding vector as a list of floats
                - text: The original text chunk
        """
        return [
            {
                "article_hash_id": self.article_hash_id,
                "chunk_id": f"{self.article_hash_id}_{hash(embedding.text)}",
                "embedding_vector": embedding.vector,
                "text": embedding.text,
            }
            for embedding in self.embeddings
        ]
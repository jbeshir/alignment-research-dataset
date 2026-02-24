from typing import List

from pydantic import BaseModel
from align_data.embeddings.embedding_utils import Embedding


class MissingEmbeddingModelError(Exception):
    pass


class PineconeEntry(BaseModel):
    hash_id: str
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

        return f"PineconeEntry(hash_id={self.hash_id!r}, text_chunks={display_chunks(self.embeddings)})"

    @property
    def chunk_num(self) -> int:
        return len(self.embeddings)

    def create_pinecone_vectors(self) -> List[dict]:
        return [
            {
                "id": f"{self.hash_id}_{hash(embedding.text)}",
                "values": embedding.vector,
                "metadata": {"hash_id": self.hash_id},
            }
            for embedding in self.embeddings
        ]

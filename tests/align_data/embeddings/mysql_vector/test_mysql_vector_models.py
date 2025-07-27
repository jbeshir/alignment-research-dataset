import pytest
from align_data.embeddings.mysql_vector.mysql_vector_models import MySQLVectorEntry
from align_data.embeddings.embedding_utils import Embedding


class TestMySQLVectorEntry:
    """Test cases for MySQLVectorEntry model."""

    def test_init_with_valid_data(self):
        """Test MySQLVectorEntry initialization with valid data."""
        embeddings = [
            Embedding(vector=[0.1, 0.2, 0.3], text="First chunk of text"),
            Embedding(vector=[0.4, 0.5, 0.6], text="Second chunk of text"),
        ]
        
        entry = MySQLVectorEntry(
            article_hash_id="test_hash_123",
            embeddings=embeddings
        )
        
        assert entry.article_hash_id == "test_hash_123"
        assert len(entry.embeddings) == 2
        assert entry.embeddings[0].text == "First chunk of text"
        assert entry.embeddings[1].text == "Second chunk of text"

    def test_init_with_empty_embeddings(self):
        """Test MySQLVectorEntry initialization with empty embeddings list."""
        entry = MySQLVectorEntry(
            article_hash_id="test_hash_123",
            embeddings=[]
        )
        
        assert entry.article_hash_id == "test_hash_123"
        assert len(entry.embeddings) == 0

    def test_chunk_num_property(self):
        """Test chunk_num property returns correct number of embeddings."""
        embeddings = [
            Embedding(vector=[0.1, 0.2], text="Chunk 1"),
            Embedding(vector=[0.3, 0.4], text="Chunk 2"),
            Embedding(vector=[0.5, 0.6], text="Chunk 3"),
        ]
        
        entry = MySQLVectorEntry(
            article_hash_id="test_hash",
            embeddings=embeddings
        )
        
        assert entry.chunk_num == 3

    def test_chunk_num_property_empty(self):
        """Test chunk_num property with empty embeddings."""
        entry = MySQLVectorEntry(
            article_hash_id="test_hash",
            embeddings=[]
        )
        
        assert entry.chunk_num == 0

    def test_create_mysql_records_basic(self):
        """Test create_mysql_records method with basic data."""
        embeddings = [
            Embedding(vector=[0.1, 0.2, 0.3], text="First chunk"),
            Embedding(vector=[0.4, 0.5, 0.6], text="Second chunk"),
        ]
        
        entry = MySQLVectorEntry(
            article_hash_id="test_hash_123",
            embeddings=embeddings
        )
        
        records = entry.create_mysql_records()
        
        assert len(records) == 2
        
        # Check first record
        first_record = records[0]
        assert first_record["article_hash_id"] == "test_hash_123"
        assert first_record["chunk_id"] == f"test_hash_123_{hash('First chunk')}"
        assert first_record["embedding_vector"] == [0.1, 0.2, 0.3]
        assert first_record["text"] == "First chunk"
        
        # Check second record
        second_record = records[1]
        assert second_record["article_hash_id"] == "test_hash_123"
        assert second_record["chunk_id"] == f"test_hash_123_{hash('Second chunk')}"
        assert second_record["embedding_vector"] == [0.4, 0.5, 0.6]
        assert second_record["text"] == "Second chunk"

    def test_create_mysql_records_empty(self):
        """Test create_mysql_records method with empty embeddings."""
        entry = MySQLVectorEntry(
            article_hash_id="test_hash_123",
            embeddings=[]
        )
        
        records = entry.create_mysql_records()
        
        assert records == []

    def test_create_mysql_records_chunk_id_uniqueness(self):
        """Test that chunk_id is unique for different text content."""
        embeddings = [
            Embedding(vector=[0.1, 0.2], text="Unique text 1"),
            Embedding(vector=[0.3, 0.4], text="Unique text 2"),
        ]
        
        entry = MySQLVectorEntry(
            article_hash_id="test_hash",
            embeddings=embeddings
        )
        
        records = entry.create_mysql_records()
        
        chunk_ids = [record["chunk_id"] for record in records]
        assert len(set(chunk_ids)) == 2  # All chunk_ids should be unique

    def test_create_mysql_records_same_text_same_chunk_id(self):
        """Test that identical text produces the same chunk_id."""
        same_text = "This is identical text"
        embeddings = [
            Embedding(vector=[0.1, 0.2], text=same_text),
            Embedding(vector=[0.3, 0.4], text=same_text),
        ]
        
        entry = MySQLVectorEntry(
            article_hash_id="test_hash",
            embeddings=embeddings
        )
        
        records = entry.create_mysql_records()
        
        chunk_ids = [record["chunk_id"] for record in records]
        assert chunk_ids[0] == chunk_ids[1]  # Same text should produce same chunk_id

    def test_create_mysql_records_preserves_vector_data(self):
        """Test that vector data is preserved correctly in MySQL records."""
        complex_vector = [0.123456789, -0.987654321, 0.0, 1.0, -1.0]
        embeddings = [
            Embedding(vector=complex_vector, text="Test text"),
        ]
        
        entry = MySQLVectorEntry(
            article_hash_id="test_hash",
            embeddings=embeddings
        )
        
        records = entry.create_mysql_records()
        
        assert records[0]["embedding_vector"] == complex_vector

    def test_repr_short_text(self):
        """Test __repr__ method with short text chunks."""
        embeddings = [
            Embedding(vector=[0.1, 0.2], text="Short text"),
        ]
        
        entry = MySQLVectorEntry(
            article_hash_id="test_hash",
            embeddings=embeddings
        )
        
        repr_str = repr(entry)
        assert "MySQLVectorEntry" in repr_str
        assert "test_hash" in repr_str
        assert "Short text" in repr_str

    def test_repr_long_text(self):
        """Test __repr__ method with long text chunks that get truncated."""
        long_text = "A" * 200  # Create text longer than 100 characters
        embeddings = [
            Embedding(vector=[0.1, 0.2], text=long_text),
        ]
        
        entry = MySQLVectorEntry(
            article_hash_id="test_hash",
            embeddings=embeddings
        )
        
        repr_str = repr(entry)
        assert "MySQLVectorEntry" in repr_str
        assert "test_hash" in repr_str
        assert "[...]" in repr_str  # Should show truncation

    def test_repr_many_chunks(self):
        """Test __repr__ method with many chunks that get truncated."""
        embeddings = [
            Embedding(vector=[0.1, 0.2], text=f"Chunk {i} text content")
            for i in range(50)  # Create many chunks
        ]
        
        entry = MySQLVectorEntry(
            article_hash_id="test_hash",
            embeddings=embeddings
        )
        
        repr_str = repr(entry)
        assert "MySQLVectorEntry" in repr_str
        assert "test_hash" in repr_str
        # Should truncate the chunks display when it gets too long

    def test_pydantic_validation(self):
        """Test that Pydantic validation works correctly."""
        # Test with invalid data types
        with pytest.raises(Exception):  # Pydantic will raise validation error
            MySQLVectorEntry(
                article_hash_id=123,  # Should be string
                embeddings=[]
            )
        
        with pytest.raises(Exception):  # Pydantic will raise validation error
            MySQLVectorEntry(
                article_hash_id="test_hash",
                embeddings="not_a_list"  # Should be list
            )
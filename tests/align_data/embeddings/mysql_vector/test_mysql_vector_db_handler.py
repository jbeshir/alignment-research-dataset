"""
Unit tests for MySQLVectorDB handler.

These tests verify the CRUD operations and error handling of the MySQLVectorDB class.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from datetime import datetime

from align_data.embeddings.mysql_vector.mysql_vector_db_handler import MySQLVectorDB
from align_data.embeddings.mysql_vector.mysql_vector_models import MySQLVectorEntry
from align_data.embeddings.embedding_utils import Embedding
from align_data.db.models import ArticleEmbedding, Article


class TestMySQLVectorDB:
    """Test suite for MySQLVectorDB class."""
    
    @pytest.fixture
    def mock_session_factory(self):
        """Create a mock session factory."""
        mock_session = Mock()
        mock_session_factory = Mock(return_value=mock_session)
        mock_session_factory.return_value.__enter__ = Mock(return_value=mock_session)
        mock_session_factory.return_value.__exit__ = Mock(return_value=None)
        return mock_session_factory
    
    @pytest.fixture
    def mysql_vector_db(self, mock_session_factory):
        """Create MySQLVectorDB instance with mocked session factory."""
        return MySQLVectorDB(session_factory=mock_session_factory)
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        return [
            Embedding(vector=[0.1, 0.2, 0.3], text="Sample text 1"),
            Embedding(vector=[0.4, 0.5, 0.6], text="Sample text 2"),
        ]
    
    @pytest.fixture
    def sample_mysql_entry(self, sample_embeddings):
        """Create sample MySQLVectorEntry for testing."""
        return MySQLVectorEntry(
            article_hash_id="test_hash_123",
            embeddings=sample_embeddings
        )
    
    def test_init_default_session_factory(self):
        """Test MySQLVectorDB initialization with default session factory."""
        from align_data.db.session import make_session
        db = MySQLVectorDB()
        assert db.session_factory == make_session
    
    def test_init_custom_session_factory(self, mock_session_factory):
        """Test MySQLVectorDB initialization with custom session factory."""
        db = MySQLVectorDB(session_factory=mock_session_factory)
        assert db.session_factory == mock_session_factory


class TestUpsertEntry:
    """Test suite for upsert_entry method."""
    
    @pytest.fixture
    def mock_session_factory(self):
        """Create a mock session factory."""
        mock_session = Mock()
        mock_session_factory = Mock(return_value=mock_session)
        mock_session_factory.return_value.__enter__ = Mock(return_value=mock_session)
        mock_session_factory.return_value.__exit__ = Mock(return_value=None)
        return mock_session_factory
    
    @pytest.fixture
    def mysql_vector_db(self, mock_session_factory):
        """Create MySQLVectorDB instance with mocked session factory."""
        return MySQLVectorDB(session_factory=mock_session_factory)
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        return [
            Embedding(vector=[0.1, 0.2, 0.3], text="Sample text 1"),
            Embedding(vector=[0.4, 0.5, 0.6], text="Sample text 2"),
        ]
    
    @pytest.fixture
    def sample_mysql_entry(self, sample_embeddings):
        """Create sample MySQLVectorEntry for testing."""
        return MySQLVectorEntry(
            article_hash_id="test_hash_123",
            embeddings=sample_embeddings
        )
    
    def test_upsert_entry_success(self, mysql_vector_db, sample_mysql_entry, mock_session_factory):
        """Test successful upsert of embeddings."""
        mock_session = mock_session_factory.return_value.__enter__.return_value
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.delete.return_value = 2  # Simulate deleting 2 existing embeddings
        
        mysql_vector_db.upsert_entry(sample_mysql_entry)
        
        # Verify delete was called
        mock_session.query.assert_called_with(ArticleEmbedding)
        mock_query.filter.assert_called()
        mock_query.delete.assert_called_once()
        
        # Verify add was called for each embedding
        assert mock_session.add.call_count == 2
        
        # Verify commit was called
        mock_session.commit.assert_called_once()
    
    def test_upsert_entry_empty_embeddings(self, mysql_vector_db, mock_session_factory):
        """Test upsert with empty embeddings list."""
        empty_entry = MySQLVectorEntry(
            article_hash_id="test_hash_123",
            embeddings=[]
        )
        
        mysql_vector_db.upsert_entry(empty_entry)
        
        # Verify no database operations were performed
        mock_session = mock_session_factory.return_value.__enter__.return_value
        mock_session.query.assert_not_called()
        mock_session.add.assert_not_called()
        mock_session.commit.assert_not_called()
    
    def test_upsert_entry_integrity_error(self, mysql_vector_db, sample_mysql_entry, mock_session_factory):
        """Test upsert with integrity error."""
        mock_session = mock_session_factory.return_value.__enter__.return_value
        mock_session.commit.side_effect = IntegrityError("statement", "params", "orig")
        
        with pytest.raises(IntegrityError):
            mysql_vector_db.upsert_entry(sample_mysql_entry)
    
    def test_upsert_entry_sqlalchemy_error(self, mysql_vector_db, sample_mysql_entry, mock_session_factory):
        """Test upsert with SQLAlchemy error."""
        mock_session = mock_session_factory.return_value.__enter__.return_value
        mock_session.commit.side_effect = SQLAlchemyError("Database error")
        
        with pytest.raises(SQLAlchemyError):
            mysql_vector_db.upsert_entry(sample_mysql_entry)
    
    def test_upsert_entry_unexpected_error(self, mysql_vector_db, sample_mysql_entry, mock_session_factory):
        """Test upsert with unexpected error."""
        mock_session = mock_session_factory.return_value.__enter__.return_value
        mock_session.commit.side_effect = Exception("Unexpected error")
        
        with pytest.raises(Exception):
            mysql_vector_db.upsert_entry(sample_mysql_entry)


class TestDeleteEntries:
    """Test suite for delete_entries method."""
    
    @pytest.fixture
    def mock_session_factory(self):
        """Create a mock session factory."""
        mock_session = Mock()
        mock_session_factory = Mock(return_value=mock_session)
        mock_session_factory.return_value.__enter__ = Mock(return_value=mock_session)
        mock_session_factory.return_value.__exit__ = Mock(return_value=None)
        return mock_session_factory
    
    @pytest.fixture
    def mysql_vector_db(self, mock_session_factory):
        """Create MySQLVectorDB instance with mocked session factory."""
        return MySQLVectorDB(session_factory=mock_session_factory)
    
    def test_delete_entries_success(self, mysql_vector_db, mock_session_factory):
        """Test successful deletion of embeddings."""
        mock_session = mock_session_factory.return_value.__enter__.return_value
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.delete.return_value = 5  # Simulate deleting 5 embeddings
        
        article_ids = ["hash1", "hash2", "hash3"]
        mysql_vector_db.delete_entries(article_ids)
        
        # Verify query and filter were called
        mock_session.query.assert_called_with(ArticleEmbedding)
        mock_query.filter.assert_called()
        mock_query.delete.assert_called_once_with(synchronize_session=False)
        
        # Verify commit was called
        mock_session.commit.assert_called_once()
    
    def test_delete_entries_empty_list(self, mysql_vector_db, mock_session_factory):
        """Test deletion with empty article IDs list."""
        mysql_vector_db.delete_entries([])
        
        # Verify no database operations were performed
        mock_session = mock_session_factory.return_value.__enter__.return_value
        mock_session.query.assert_not_called()
        mock_session.commit.assert_not_called()
    
    def test_delete_entries_sqlalchemy_error(self, mysql_vector_db, mock_session_factory):
        """Test deletion with SQLAlchemy error."""
        mock_session = mock_session_factory.return_value.__enter__.return_value
        mock_session.commit.side_effect = SQLAlchemyError("Database error")
        
        with pytest.raises(SQLAlchemyError):
            mysql_vector_db.delete_entries(["hash1", "hash2"])
    
    def test_delete_entries_unexpected_error(self, mysql_vector_db, mock_session_factory):
        """Test deletion with unexpected error."""
        mock_session = mock_session_factory.return_value.__enter__.return_value
        mock_session.commit.side_effect = Exception("Unexpected error")
        
        with pytest.raises(Exception):
            mysql_vector_db.delete_entries(["hash1", "hash2"])


class TestQuerySimilarVectors:
    """Test suite for query_similar_vectors method."""
    
    @pytest.fixture
    def mock_session_factory(self):
        """Create a mock session factory."""
        mock_session = Mock()
        mock_session_factory = Mock(return_value=mock_session)
        mock_session_factory.return_value.__enter__ = Mock(return_value=mock_session)
        mock_session_factory.return_value.__exit__ = Mock(return_value=None)
        return mock_session_factory
    
    @pytest.fixture
    def mysql_vector_db(self, mock_session_factory):
        """Create MySQLVectorDB instance with mocked session factory."""
        return MySQLVectorDB(session_factory=mock_session_factory)
    
    @pytest.fixture
    def mock_query_results(self):
        """Create mock query results."""
        mock_result = Mock()
        mock_result.chunk_id = "chunk_123"
        mock_result.article_hash_id = "hash_123"
        mock_result.text = "Sample text"
        mock_result.article_title = "Sample Title"
        mock_result.article_url = "http://example.com"
        mock_result.article_source = "test_source"
        mock_result.article_authors = "Test Author"
        mock_result.date_published = datetime(2023, 1, 1)
        mock_result.similarity_score = 0.95
        return [mock_result]
    
    def test_query_similar_vectors_success(self, mysql_vector_db, mock_session_factory, mock_query_results):
        """Test successful vector similarity query."""
        mock_session = mock_session_factory.return_value.__enter__.return_value
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = mock_query_results
        
        query_vector = [0.1, 0.2, 0.3]
        results = mysql_vector_db.query_similar_vectors(query_vector, top_k=5)
        
        # Verify query was built correctly
        mock_session.query.assert_called()
        mock_query.join.assert_called()
        mock_query.filter.assert_called()
        mock_query.order_by.assert_called()
        mock_query.limit.assert_called_with(5)
        mock_query.all.assert_called_once()
        
        # Verify results format
        assert len(results) == 1
        result = results[0]
        assert result['chunk_id'] == "chunk_123"
        assert result['article_hash_id'] == "hash_123"
        assert result['text'] == "Sample text"
        assert result['article_title'] == "Sample Title"
        assert result['similarity_score'] == 0.95
        
    def test_query_similar_vectors_mysql_vector_distance(self, mysql_vector_db, mock_session_factory):
        """Test that MySQL VECTOR_DISTANCE function is used correctly."""
        mock_session = mock_session_factory.return_value.__enter__.return_value
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        
        query_vector = [0.1, 0.2, 0.3]
        mysql_vector_db.query_similar_vectors(query_vector, top_k=5)
        
        # Verify that the query includes the vector similarity calculation
        # The exact SQL text should include VECTOR_DISTANCE function
        mock_session.query.assert_called()
        call_args = mock_session.query.call_args[0]
        
        # Check that one of the query columns contains VECTOR_DISTANCE SQL
        found_vector_distance = False
        for arg in call_args:
            if hasattr(arg, 'text') and 'VECTOR_DISTANCE' in str(arg.text):
                found_vector_distance = True
                break
        
        assert found_vector_distance, "Query should include VECTOR_DISTANCE function"
    
    def test_query_similar_vectors_with_exclusions(self, mysql_vector_db, mock_session_factory, mock_query_results):
        """Test vector similarity query with exclusions."""
        mock_session = mock_session_factory.return_value.__enter__.return_value
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = mock_query_results
        
        query_vector = [0.1, 0.2, 0.3]
        exclude_ids = ["hash_456", "hash_789"]
        results = mysql_vector_db.query_similar_vectors(
            query_vector, 
            top_k=5, 
            exclude_hash_ids=exclude_ids
        )
        
        # Verify filter was called twice (once for is_valid, once for exclusions)
        assert mock_query.filter.call_count == 2
        mock_query.all.assert_called_once()
        assert len(results) == 1
    
    def test_query_similar_vectors_empty_vector(self, mysql_vector_db, mock_session_factory):
        """Test query with empty vector."""
        results = mysql_vector_db.query_similar_vectors([])
        
        # Verify no database operations were performed
        mock_session = mock_session_factory.return_value.__enter__.return_value
        mock_session.query.assert_not_called()
        assert results == []
    
    def test_query_similar_vectors_sqlalchemy_error(self, mysql_vector_db, mock_session_factory):
        """Test query with SQLAlchemy error."""
        mock_session = mock_session_factory.return_value.__enter__.return_value
        mock_session.query.side_effect = SQLAlchemyError("Database error")
        
        with pytest.raises(SQLAlchemyError):
            mysql_vector_db.query_similar_vectors([0.1, 0.2, 0.3])


class TestGetArticleAverageEmbedding:
    """Test suite for get_article_average_embedding method."""
    
    @pytest.fixture
    def mock_session_factory(self):
        """Create a mock session factory."""
        mock_session = Mock()
        mock_session_factory = Mock(return_value=mock_session)
        mock_session_factory.return_value.__enter__ = Mock(return_value=mock_session)
        mock_session_factory.return_value.__exit__ = Mock(return_value=None)
        return mock_session_factory
    
    @pytest.fixture
    def mysql_vector_db(self, mock_session_factory):
        """Create MySQLVectorDB instance with mocked session factory."""
        return MySQLVectorDB(session_factory=mock_session_factory)
    
    def test_get_article_average_embedding_success(self, mysql_vector_db, mock_session_factory):
        """Test successful calculation of average embedding."""
        mock_session = mock_session_factory.return_value.__enter__.return_value
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        
        # Mock embedding results
        mock_embedding1 = Mock()
        mock_embedding1.embedding_vector = "[0.1, 0.2, 0.3]"
        mock_embedding2 = Mock()
        mock_embedding2.embedding_vector = "[0.4, 0.5, 0.6]"
        mock_query.all.return_value = [mock_embedding1, mock_embedding2]
        
        result = mysql_vector_db.get_article_average_embedding("test_hash")
        
        # Verify query was called correctly
        mock_session.query.assert_called()
        mock_query.filter.assert_called()
        mock_query.all.assert_called_once()
        
        # Verify average calculation
        expected_average = [0.25, 0.35, 0.45]  # Average of [0.1,0.2,0.3] and [0.4,0.5,0.6]
        assert len(result) == len(expected_average)
        for i, (actual, expected) in enumerate(zip(result, expected_average)):
            assert abs(actual - expected) < 1e-10, f"Mismatch at index {i}: {actual} != {expected}"
    
    def test_get_article_average_embedding_single_vector(self, mysql_vector_db, mock_session_factory):
        """Test average calculation with single embedding."""
        mock_session = mock_session_factory.return_value.__enter__.return_value
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        
        # Mock single embedding result
        mock_embedding = Mock()
        mock_embedding.embedding_vector = "[0.1, 0.2, 0.3]"
        mock_query.all.return_value = [mock_embedding]
        
        result = mysql_vector_db.get_article_average_embedding("test_hash")
        
        # Should return the single vector as-is
        assert result == [0.1, 0.2, 0.3]
    
    def test_get_article_average_embedding_no_embeddings(self, mysql_vector_db, mock_session_factory):
        """Test average calculation with no embeddings found."""
        mock_session = mock_session_factory.return_value.__enter__.return_value
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = []
        
        result = mysql_vector_db.get_article_average_embedding("test_hash")
        
        assert result is None
    
    def test_get_article_average_embedding_empty_hash_id(self, mysql_vector_db, mock_session_factory):
        """Test average calculation with empty hash ID."""
        result = mysql_vector_db.get_article_average_embedding("")
        
        # Verify no database operations were performed
        mock_session = mock_session_factory.return_value.__enter__.return_value
        mock_session.query.assert_not_called()
        assert result is None
    
    def test_get_article_average_embedding_invalid_vector_format(self, mysql_vector_db, mock_session_factory):
        """Test average calculation with invalid vector format."""
        mock_session = mock_session_factory.return_value.__enter__.return_value
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        
        # Mock embedding with invalid format
        mock_embedding = Mock()
        mock_embedding.embedding_vector = "invalid_format"
        mock_query.all.return_value = [mock_embedding]
        
        result = mysql_vector_db.get_article_average_embedding("test_hash")
        
        assert result is None
    
    def test_get_article_average_embedding_inconsistent_dimensions(self, mysql_vector_db, mock_session_factory):
        """Test average calculation with inconsistent vector dimensions."""
        mock_session = mock_session_factory.return_value.__enter__.return_value
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        
        # Mock embeddings with different dimensions
        mock_embedding1 = Mock()
        mock_embedding1.embedding_vector = "[0.1, 0.2, 0.3]"
        mock_embedding2 = Mock()
        mock_embedding2.embedding_vector = "[0.4, 0.5]"  # Different dimension
        mock_query.all.return_value = [mock_embedding1, mock_embedding2]
        
        result = mysql_vector_db.get_article_average_embedding("test_hash")
        
        # Should return None due to inconsistent dimensions
        assert result is None
    
    def test_get_article_average_embedding_sqlalchemy_error(self, mysql_vector_db, mock_session_factory):
        """Test average calculation with SQLAlchemy error."""
        mock_session = mock_session_factory.return_value.__enter__.return_value
        mock_session.query.side_effect = SQLAlchemyError("Database error")
        
        with pytest.raises(SQLAlchemyError):
            mysql_vector_db.get_article_average_embedding("test_hash")


class TestParseVectorString:
    """Test suite for _parse_vector_string method."""
    
    @pytest.fixture
    def mysql_vector_db(self):
        """Create MySQLVectorDB instance for testing."""
        return MySQLVectorDB()
    
    def test_parse_vector_string_python_list_format(self, mysql_vector_db):
        """Test parsing Python list format string."""
        vector_str = "[0.1, 0.2, 0.3]"
        result = mysql_vector_db._parse_vector_string(vector_str)
        assert result == [0.1, 0.2, 0.3]
    
    def test_parse_vector_string_json_format(self, mysql_vector_db):
        """Test parsing JSON array format string."""
        vector_str = '[0.1, 0.2, 0.3]'
        result = mysql_vector_db._parse_vector_string(vector_str)
        assert result == [0.1, 0.2, 0.3]
    
    def test_parse_vector_string_already_list(self, mysql_vector_db):
        """Test parsing when input is already a list."""
        vector_list = [0.1, 0.2, 0.3]
        result = mysql_vector_db._parse_vector_string(vector_list)
        assert result == [0.1, 0.2, 0.3]
    
    def test_parse_vector_string_empty_input(self, mysql_vector_db):
        """Test parsing empty input."""
        result = mysql_vector_db._parse_vector_string("")
        assert result is None
        
        result = mysql_vector_db._parse_vector_string(None)
        assert result is None
    
    def test_parse_vector_string_invalid_format(self, mysql_vector_db):
        """Test parsing invalid format."""
        result = mysql_vector_db._parse_vector_string("invalid_format")
        assert result is None
        
        result = mysql_vector_db._parse_vector_string("[0.1, 'invalid']")
        assert result is None
    
    def test_parse_vector_string_mixed_types(self, mysql_vector_db):
        """Test parsing with mixed int/float types."""
        vector_str = "[1, 2.5, 3]"
        result = mysql_vector_db._parse_vector_string(vector_str)
        assert result == [1.0, 2.5, 3.0]
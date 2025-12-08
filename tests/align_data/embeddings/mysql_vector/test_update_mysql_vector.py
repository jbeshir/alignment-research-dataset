"""
Unit tests for MySQL Vector Action base class functionality.

This module tests the MySQLVectorAction base class and its helper functions,
ensuring proper batch processing, error handling, and session management.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from align_data.db.models import Article
from align_data.embeddings.mysql_vector.update_mysql_vector import (
    MySQLVectorAction,
    get_text_chunks,
    get_authors_str
)
from align_data.embeddings.mysql_vector.mysql_vector_db_handler import MySQLVectorDB
from align_data.embeddings.mysql_vector.mysql_vector_models import MySQLVectorEntry


class TestMySQLVectorAction:
    """Test cases for MySQLVectorAction base class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_mysql_db = Mock(spec=MySQLVectorDB)
        self.action = MySQLVectorAction(mysql_db=self.mock_mysql_db)

    def test_init_with_mysql_db(self):
        """Test initialization with provided MySQL DB instance."""
        action = MySQLVectorAction(mysql_db=self.mock_mysql_db)
        assert action.mysql_db is self.mock_mysql_db
        assert action.batch_size == 10

    def test_init_without_mysql_db(self):
        """Test initialization without MySQL DB instance creates new one."""
        with patch('align_data.embeddings.mysql_vector.update_mysql_vector.MySQLVectorDB') as mock_db_class:
            mock_db_instance = Mock()
            mock_db_class.return_value = mock_db_instance
            
            action = MySQLVectorAction()
            
            mock_db_class.assert_called_once_with()
            assert action.mysql_db is mock_db_instance

    def test_articles_by_source_not_implemented(self):
        """Test that _articles_by_source raises NotImplementedError."""
        session = Mock(spec=Session)
        sources = ["test_source"]
        
        with pytest.raises(NotImplementedError):
            self.action._articles_by_source(session, sources, False)

    def test_articles_by_id_not_implemented(self):
        """Test that _articles_by_id raises NotImplementedError."""
        session = Mock(spec=Session)
        ids = ["test_id"]
        
        with pytest.raises(NotImplementedError):
            self.action._articles_by_id(session, ids, False)

    def test_process_batch_not_implemented(self):
        """Test that process_batch raises NotImplementedError."""
        batch = []
        
        with pytest.raises(NotImplementedError):
            self.action.process_batch(batch)

    def test_batch_entries(self):
        """Test batch_entries method yields articles in correct batch sizes."""
        # Create mock articles
        articles = [Mock(spec=Article) for _ in range(25)]
        
        def article_generator():
            for article in articles:
                yield article
        
        batches = list(self.action.batch_entries(article_generator()))
        
        # Should have 3 batches: 10, 10, 5
        assert len(batches) == 3
        assert len(batches[0]) == 10
        assert len(batches[1]) == 10
        assert len(batches[2]) == 5
        
        # Verify all articles are included
        all_batched_articles = []
        for batch in batches:
            all_batched_articles.extend(batch)
        assert len(all_batched_articles) == 25

    def test_batch_entries_empty_stream(self):
        """Test batch_entries with empty article stream."""
        def empty_generator():
            return
            yield  # This line is never reached
        
        batches = list(self.action.batch_entries(empty_generator()))
        assert len(batches) == 0

    def test_batch_entries_single_batch(self):
        """Test batch_entries with fewer articles than batch size."""
        articles = [Mock(spec=Article) for _ in range(5)]
        
        def article_generator():
            for article in articles:
                yield article
        
        batches = list(self.action.batch_entries(article_generator()))
        
        assert len(batches) == 1
        assert len(batches[0]) == 5

    @patch('align_data.embeddings.mysql_vector.update_mysql_vector.logger')
    def test_save_batch_success(self, mock_logger):
        """Test save_batch with successful processing."""
        session = Mock(spec=Session)
        batch = [Mock(), Mock()]
        processed_articles = [Mock(spec=Article), Mock(spec=Article)]
        
        # Mock process_batch to return processed articles
        self.action.process_batch = Mock(return_value=processed_articles)
        
        self.action.save_batch(session, batch)
        
        self.action.process_batch.assert_called_once_with(batch)
        session.add_all.assert_called_once_with(processed_articles)
        session.commit.assert_called_once()
        session.rollback.assert_not_called()

    @patch('align_data.embeddings.mysql_vector.update_mysql_vector.logger')
    @patch('align_data.embeddings.mysql_vector.update_mysql_vector.traceback')
    def test_save_batch_error_handling(self, mock_traceback, mock_logger):
        """Test save_batch error handling and rollback."""
        session = Mock(spec=Session)
        batch = [Mock(), Mock()]
        
        # Mock process_batch to raise an exception
        test_error = SQLAlchemyError("Test database error")
        self.action.process_batch = Mock(side_effect=test_error)
        
        # Should not raise exception, but handle it gracefully
        self.action.save_batch(session, batch)
        
        self.action.process_batch.assert_called_once_with(batch)
        session.rollback.assert_called_once()
        session.commit.assert_not_called()
        mock_logger.error.assert_called_once()
        mock_traceback.print_exc.assert_called_once()

    @patch('align_data.embeddings.mysql_vector.update_mysql_vector.make_session')
    def test_update_with_logging(self, mock_make_session):
        """Test _update_with_logging method."""
        # Setup mocks
        session = Mock(spec=Session)
        articles_query = Mock()
        articles_query.count.return_value = 25
        
        # Mock articles for batching
        articles = [Mock(spec=Article) for _ in range(25)]
        
        def article_generator():
            for article in articles:
                yield article
        
        # Mock batch_entries and save_batch
        self.action.batch_entries = Mock(return_value=[
            articles[:10], articles[10:20], articles[20:25]
        ])
        self.action.save_batch = Mock()
        
        with patch('align_data.embeddings.mysql_vector.update_mysql_vector.logger') as mock_logger:
            self.action._update_with_logging(session, articles_query, log_progress=True)
        
        # Verify logging
        mock_logger.info.assert_has_calls([
            call("Processing %s items", 25),
            call("Progress: %.1f%% (%d/%d)", 40.0, 10, 25),
            call("Progress: %.1f%% (%d/%d)", 80.0, 20, 25),
            call("Progress: %.1f%% (%d/%d)", 100.0, 25, 25),
            call("Completed processing %s items", 25)
        ])
        
        # Verify batch processing
        self.action.batch_entries.assert_called_once_with(articles_query)
        assert self.action.save_batch.call_count == 3

    @patch('align_data.embeddings.mysql_vector.update_mysql_vector.make_session')
    def test_update_with_logging_no_progress(self, mock_make_session):
        """Test _update_with_logging without progress logging."""
        session = Mock(spec=Session)
        articles_query = Mock()
        articles_query.count.return_value = 5
        
        self.action.batch_entries = Mock(return_value=[[Mock()]])
        self.action.save_batch = Mock()
        
        with patch('align_data.embeddings.mysql_vector.update_mysql_vector.logger') as mock_logger:
            self.action._update_with_logging(session, articles_query, log_progress=False)
        
        # Should not log progress
        mock_logger.info.assert_not_called()

    @patch('align_data.embeddings.mysql_vector.update_mysql_vector.make_session')
    def test_update(self, mock_make_session):
        """Test update method."""
        session = Mock(spec=Session)
        mock_make_session.return_value.__enter__.return_value = session
        
        articles_query = Mock()
        sources = ["test_source"]
        
        # Mock _articles_by_source and _update_with_logging
        self.action._articles_by_source = Mock(return_value=articles_query)
        self.action._update_with_logging = Mock()
        
        self.action.update(sources, force_update=True, log_progress=False)
        
        self.action._articles_by_source.assert_called_once_with(session, sources, True)
        self.action._update_with_logging.assert_called_once_with(session, articles_query, False)

    @patch('align_data.embeddings.mysql_vector.update_mysql_vector.make_session')
    def test_update_articles_by_ids(self, mock_make_session):
        """Test update_articles_by_ids method."""
        session = Mock(spec=Session)
        mock_make_session.return_value.__enter__.return_value = session
        
        articles_query = Mock()
        hash_ids = ["id1", "id2"]
        
        # Mock _articles_by_id and _update_with_logging
        self.action._articles_by_id = Mock(return_value=articles_query)
        self.action._update_with_logging = Mock()
        
        self.action.update_articles_by_ids(hash_ids, force_update=False, log_progress=True)
        
        self.action._articles_by_id.assert_called_once_with(session, hash_ids, False)
        self.action._update_with_logging.assert_called_once_with(session, articles_query, True)


class TestHelperFunctions:
    """Test cases for helper functions."""

    def test_get_authors_str_empty_list(self):
        """Test get_authors_str with empty list."""
        result = get_authors_str([])
        assert result == "n/a"

    def test_get_authors_str_single_author(self):
        """Test get_authors_str with single author."""
        result = get_authors_str(["John Doe"])
        assert result == "John Doe"

    def test_get_authors_str_two_authors(self):
        """Test get_authors_str with two authors."""
        result = get_authors_str(["John Doe", "Jane Smith"])
        assert result == "John Doe and Jane Smith"

    def test_get_authors_str_multiple_authors(self):
        """Test get_authors_str with multiple authors."""
        result = get_authors_str(["John Doe", "Jane Smith", "Bob Johnson"])
        assert result == "John Doe, Jane Smith and Bob Johnson"

    def test_get_authors_str_many_authors_truncated(self):
        """Test get_authors_str with more than 4 authors (should truncate to 4)."""
        authors = ["Author1", "Author2", "Author3", "Author4", "Author5", "Author6"]
        result = get_authors_str(authors)
        assert result == "Author1, Author2, Author3 and Author4"

    def test_get_authors_str_newline_replacement(self):
        """Test get_authors_str replaces newlines with spaces."""
        result = get_authors_str(["John\nDoe", "Jane\nSmith"])
        assert result == "John Doe and Jane Smith"

    def test_get_authors_str_long_string_truncation(self):
        """Test get_authors_str truncates very long author strings."""
        long_author = "A" * 600  # Very long author name
        result = get_authors_str([long_author])
        assert len(result) == 500
        assert result.endswith("...")

    def test_get_text_chunks(self):
        """Test get_text_chunks function."""
        # Create mock article
        article = Mock(spec=Article)
        article.title = "Test Article\nTitle"
        article.authors = "John Doe, Jane Smith"
        article.text = "This is the main article text."
        
        # Mock summary
        mock_summary = Mock()
        mock_summary.text = "This is a summary."
        article.summaries = [mock_summary]
        
        with patch('align_data.embeddings.mysql_vector.update_mysql_vector.split_text') as mock_split:
            mock_split.side_effect = [
                ["chunk1", "chunk2"],  # For article.text
                ["summary_chunk"]      # For summary.text
            ]
            
            result = get_text_chunks(article)
            
            # Verify split_text was called correctly
            assert mock_split.call_count == 2
            mock_split.assert_has_calls([
                call("This is the main article text."),
                call("This is a summary.")
            ])
            
            # Verify result format
            expected_signature = "Title: Test Article Title; Author(s): John Doe and Jane Smith."
            expected_chunks = [
                f'###{expected_signature}###\n"""chunk1"""',
                f'###{expected_signature}###\n"""chunk2"""',
                f'###{expected_signature}###\n"""summary_chunk"""'
            ]
            
            assert result == expected_chunks

    def test_get_text_chunks_no_summaries(self):
        """Test get_text_chunks with article that has no summaries."""
        article = Mock(spec=Article)
        article.title = "Test Article"
        article.authors = "John Doe"
        article.text = "Article text."
        article.summaries = []
        
        with patch('align_data.embeddings.mysql_vector.update_mysql_vector.split_text') as mock_split:
            mock_split.return_value = ["chunk1"]
            
            result = get_text_chunks(article)
            
            mock_split.assert_called_once_with("Article text.")
            assert len(result) == 1
            assert "Title: Test Article; Author(s): John Doe." in result[0]
            assert "chunk1" in result[0]


class TestMySQLVectorAdder:
    """Test cases for MySQLVectorAdder class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_mysql_db = Mock(spec=MySQLVectorDB)
        
        # Import here to avoid circular imports in test setup
        from align_data.embeddings.mysql_vector.update_mysql_vector import MySQLVectorAdder
        self.adder = MySQLVectorAdder(mysql_db=self.mock_mysql_db)

    def test_init(self):
        """Test MySQLVectorAdder initialization."""
        from align_data.embeddings.mysql_vector.update_mysql_vector import MySQLVectorAdder
        adder = MySQLVectorAdder(mysql_db=self.mock_mysql_db)
        assert adder.mysql_db is self.mock_mysql_db
        assert adder.batch_size == 10

    @patch('align_data.db.session.get_mysql_vector_articles_by_sources')
    def test_articles_by_source(self, mock_get_articles):
        """Test _articles_by_source method."""
        session = Mock(spec=Session)
        sources = ["test_source"]
        mock_articles = [Mock(spec=Article)]
        mock_get_articles.return_value = mock_articles
        
        result = self.adder._articles_by_source(session, sources, force_update=True)
        
        mock_get_articles.assert_called_once_with(session, sources, True)
        assert result is mock_articles

    @patch('align_data.db.session.get_mysql_vector_articles_by_ids')
    def test_articles_by_id(self, mock_get_articles):
        """Test _articles_by_id method."""
        session = Mock(spec=Session)
        ids = ["id1", "id2"]
        mock_articles = [Mock(spec=Article)]
        mock_get_articles.return_value = mock_articles
        
        result = self.adder._articles_by_id(session, ids, force_update=False)
        
        mock_get_articles.assert_called_once_with(session, ids, False)
        assert result is mock_articles

    @patch('align_data.embeddings.mysql_vector.update_mysql_vector.logger')
    def test_process_batch_success(self, mock_logger):
        """Test process_batch with successful processing."""
        from align_data.db.models import PineconeStatus
        
        # Create mock articles and entries
        article1 = Mock(spec=Article)
        article1.id = "article1"
        article2 = Mock(spec=Article)
        article2.id = "article2"
        
        entry1 = Mock(spec=MySQLVectorEntry)
        entry2 = Mock(spec=MySQLVectorEntry)
        
        batch = [(article1, entry1), (article2, entry2)]
        
        result = self.adder.process_batch(batch)
        
        # Verify MySQL DB operations
        self.mock_mysql_db.upsert_entry.assert_has_calls([
            call(entry1), call(entry2)
        ])
        
        # Verify article status updates
        assert article1.pinecone_status == PineconeStatus.added
        assert article2.pinecone_status == PineconeStatus.added
        
        # Verify return value
        assert result == [article1, article2]
        
        # Verify logging
        mock_logger.info.assert_has_calls([
            call("Processing batch of %s items", 2),
            call("Successfully processed embeddings for article article1"),
            call("Successfully processed embeddings for article article2")
        ])

    @patch('align_data.embeddings.mysql_vector.update_mysql_vector.logger')
    def test_process_batch_with_none_entry(self, mock_logger):
        """Test process_batch with None entry."""
        from align_data.db.models import PineconeStatus
        
        article = Mock(spec=Article)
        article.id = "article1"
        
        batch = [(article, None)]
        
        result = self.adder.process_batch(batch)
        
        # Should not call upsert_entry
        self.mock_mysql_db.upsert_entry.assert_not_called()
        
        # Should still update status
        assert article.pinecone_status == PineconeStatus.added
        assert result == [article]
        
        # Should log warning
        mock_logger.warning.assert_called_once_with("No MySQL entry generated for article article1")

    @patch('align_data.embeddings.mysql_vector.update_mysql_vector.logger')
    def test_process_batch_with_error(self, mock_logger):
        """Test process_batch with database error."""
        article1 = Mock(spec=Article)
        article1.id = "article1"
        article2 = Mock(spec=Article)
        article2.id = "article2"
        
        entry1 = Mock(spec=MySQLVectorEntry)
        entry2 = Mock(spec=MySQLVectorEntry)
        
        # Make first upsert fail
        self.mock_mysql_db.upsert_entry.side_effect = [Exception("DB Error"), None]
        
        batch = [(article1, entry1), (article2, entry2)]
        
        result = self.adder.process_batch(batch)
        
        # Should only return successfully processed articles
        assert result == [article2]
        
        # Should log error for failed article
        mock_logger.error.assert_called_once()
        # Check that error was logged with correct article ID
        error_call_args = mock_logger.error.call_args[0]
        assert "article1" in error_call_args[0]

    def test_batch_entries(self):
        """Test batch_entries method."""
        articles = [Mock(spec=Article) for _ in range(5)]
        
        def article_generator():
            for article in articles:
                yield article
        
        # Mock _make_mysql_entry
        mock_entries = [Mock(spec=MySQLVectorEntry) for _ in range(5)]
        self.adder._make_mysql_entry = Mock(side_effect=mock_entries)
        
        batches = list(self.adder.batch_entries(article_generator()))
        
        assert len(batches) == 1
        assert len(batches[0]) == 5
        
        # Verify each article was paired with its entry
        for i, (article, entry) in enumerate(batches[0]):
            assert article is articles[i]
            assert entry is mock_entries[i]

    @patch('align_data.embeddings.mysql_vector.update_mysql_vector.get_embeddings')
    @patch('align_data.embeddings.mysql_vector.update_mysql_vector.get_text_chunks')
    @patch('align_data.embeddings.mysql_vector.update_mysql_vector.logger')
    def test_make_mysql_entry_success(self, mock_logger, mock_get_chunks, mock_get_embeddings):
        """Test _make_mysql_entry with successful processing."""
        from align_data.embeddings.embedding_utils import Embedding
        
        # Setup mocks
        article = Mock(spec=Article)
        article.id = "test_article"
        article.title = "Test Article"
        article.comments = "initial comments"
        article.append_comment = Mock()
        
        mock_get_chunks.return_value = ["chunk1", "chunk2"]
        mock_embeddings = [
            Embedding(text="chunk1", vector=[0.1, 0.2]),
            Embedding(text="chunk2", vector=[0.3, 0.4])
        ]
        mock_get_embeddings.return_value = (mock_embeddings, [])
        
        result = self.adder._make_mysql_entry(article)
        
        # Verify function calls
        mock_get_chunks.assert_called_once_with(article)
        mock_get_embeddings.assert_called_once_with(["chunk1", "chunk2"])
        
        # Verify result
        assert result is not None
        assert result.article_hash_id == "test_article"
        assert result.embeddings == mock_embeddings
        
        # Verify article comments were cleared
        assert article.comments == ""
        
        # Verify logging
        mock_logger.info.assert_called_once_with("Getting embeddings for Test Article")

    @patch('align_data.embeddings.mysql_vector.update_mysql_vector.get_embeddings')
    @patch('align_data.embeddings.mysql_vector.update_mysql_vector.get_text_chunks')
    @patch('align_data.embeddings.mysql_vector.update_mysql_vector.logger')
    def test_make_mysql_entry_no_embeddings(self, mock_logger, mock_get_chunks, mock_get_embeddings):
        """Test _make_mysql_entry with no embeddings returned."""
        article = Mock(spec=Article)
        article.id = "test_article"
        article.title = "Test Article"
        article.comments = ""
        
        mock_get_chunks.return_value = ["chunk1"]
        mock_get_embeddings.return_value = ([], ["moderation_result"])
        
        result = self.adder._make_mysql_entry(article)
        
        assert result is None
        
        # Verify warning logs
        mock_logger.warning.assert_has_calls([
            call("No embeddings found for Test Article"),
            call("Moderation results: ['moderation_result']")
        ])
        mock_logger.info.assert_has_calls([
            call("Getting embeddings for Test Article"),
            call("text: %s", ["chunk1"])
        ])

    @patch('align_data.embeddings.mysql_vector.update_mysql_vector.get_embeddings')
    @patch('align_data.embeddings.mysql_vector.update_mysql_vector.get_text_chunks')
    @patch('align_data.embeddings.mysql_vector.update_mysql_vector.logger')
    def test_make_mysql_entry_with_moderation_flags(self, mock_logger, mock_get_chunks, mock_get_embeddings):
        """Test _make_mysql_entry with moderation flags."""
        from align_data.embeddings.embedding_utils import Embedding
        
        article = Mock(spec=Article)
        article.id = "test_article"
        article.title = "Test Article"
        article.comments = ""
        article.append_comment = Mock()
        
        mock_get_chunks.return_value = ["chunk1"]
        mock_embeddings = [Embedding(text="chunk1", vector=[0.1, 0.2])]
        moderation_results = [{"text": "flagged_chunk"}]
        mock_get_embeddings.return_value = (mock_embeddings, moderation_results)
        
        result = self.adder._make_mysql_entry(article)
        
        assert result is not None
        assert result.article_hash_id == "test_article"
        
        # Verify moderation handling
        mock_logger.warning.assert_called_once_with(
            "OpenAI moderation flagged text chunks for the following article: test_article"
        )
        article.append_comment.assert_called_once_with(
            "OpenAI moderation flagged the following text chunks: ['flagged_chunk']"
        )

    @patch('align_data.embeddings.mysql_vector.update_mysql_vector.get_embeddings')
    @patch('align_data.embeddings.mysql_vector.update_mysql_vector.get_text_chunks')
    @patch('align_data.embeddings.mysql_vector.update_mysql_vector.logger')
    def test_make_mysql_entry_validation_error(self, mock_logger, mock_get_chunks, mock_get_embeddings):
        """Test _make_mysql_entry with ValidationError."""
        from pydantic import ValidationError
        
        article = Mock(spec=Article)
        article.id = "test_article"
        article.title = "Test Article"
        article.comments = ""
        article.append_comment = Mock()
        
        mock_get_chunks.return_value = ["chunk1"]
        # Create a proper ValidationError
        validation_error = ValidationError.from_exception_data("MySQLVectorEntry", [])
        mock_get_embeddings.side_effect = validation_error
        
        result = self.adder._make_mysql_entry(article)
        
        assert result is None
        
        # Verify error handling
        mock_logger.warning.assert_called_once()
        article.append_comment.assert_called_once()

    @patch('align_data.embeddings.mysql_vector.update_mysql_vector.get_embeddings')
    @patch('align_data.embeddings.mysql_vector.update_mysql_vector.get_text_chunks')
    @patch('align_data.embeddings.mysql_vector.update_mysql_vector.logger')
    @patch('align_data.embeddings.mysql_vector.update_mysql_vector.traceback')
    def test_make_mysql_entry_unexpected_error(self, mock_traceback, mock_logger, mock_get_chunks, mock_get_embeddings):
        """Test _make_mysql_entry with unexpected error."""
        article = Mock(spec=Article)
        article.id = "test_article"
        article.title = "Test Article"
        article.comments = ""
        article.append_comment = Mock()
        
        mock_get_chunks.side_effect = Exception("Unexpected error")
        
        result = self.adder._make_mysql_entry(article)
        
        assert result is None
        
        # Verify error handling
        mock_logger.error.assert_called_once()
        mock_traceback.print_exc.assert_called_once()
        article.append_comment.assert_called_once()


class TestMySQLVectorDeleter:
    """Test cases for MySQLVectorDeleter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_mysql_db = Mock(spec=MySQLVectorDB)
        
        # Import here to avoid circular imports in test setup
        from align_data.embeddings.mysql_vector.update_mysql_vector import MySQLVectorDeleter
        self.deleter = MySQLVectorDeleter(mysql_db=self.mock_mysql_db)

    def test_init(self):
        """Test MySQLVectorDeleter initialization."""
        from align_data.embeddings.mysql_vector.update_mysql_vector import MySQLVectorDeleter
        deleter = MySQLVectorDeleter(mysql_db=self.mock_mysql_db)
        assert deleter.mysql_db is self.mock_mysql_db
        assert deleter.batch_size == 10

    @patch('align_data.db.session.get_mysql_vector_to_delete_by_sources')
    def test_articles_by_source(self, mock_get_articles):
        """Test _articles_by_source method."""
        session = Mock(spec=Session)
        sources = ["test_source"]
        mock_articles = [Mock(spec=Article)]
        mock_get_articles.return_value = mock_articles
        
        result = self.deleter._articles_by_source(session, sources, force_update=True)
        
        mock_get_articles.assert_called_once_with(session, sources)
        assert result is mock_articles

    @patch('align_data.db.session.get_mysql_vector_to_delete_by_ids')
    def test_articles_by_id(self, mock_get_articles):
        """Test _articles_by_id method."""
        session = Mock(spec=Session)
        ids = ["id1", "id2"]
        mock_articles = [Mock(spec=Article)]
        mock_get_articles.return_value = mock_articles
        
        result = self.deleter._articles_by_id(session, ids, force_update=False)
        
        mock_get_articles.assert_called_once_with(session, ids)
        assert result is mock_articles

    @patch('align_data.embeddings.mysql_vector.update_mysql_vector.logger')
    def test_process_batch_success(self, mock_logger):
        """Test process_batch with successful deletion."""
        from align_data.db.models import PineconeStatus
        
        # Create mock articles
        article1 = Mock(spec=Article)
        article1.id = "article1"
        article2 = Mock(spec=Article)
        article2.id = "article2"
        
        batch = [article1, article2]
        
        result = self.deleter.process_batch(batch)
        
        # Verify MySQL DB deletion operation
        self.mock_mysql_db.delete_entries.assert_called_once_with(["article1", "article2"])
        
        # Verify article status updates
        assert article1.pinecone_status == PineconeStatus.absent
        assert article2.pinecone_status == PineconeStatus.absent
        
        # Verify return value
        assert result == [article1, article2]
        
        # Verify logging
        mock_logger.info.assert_has_calls([
            call("Processing deletion batch of %s items", 2),
            call("Successfully deleted embeddings for article article1"),
            call("Successfully deleted embeddings for article article2")
        ])

    @patch('align_data.embeddings.mysql_vector.update_mysql_vector.logger')
    def test_process_batch_with_error(self, mock_logger):
        """Test process_batch with database error."""
        article1 = Mock(spec=Article)
        article1.id = "article1"
        article2 = Mock(spec=Article)
        article2.id = "article2"
        
        # Make deletion fail
        self.mock_mysql_db.delete_entries.side_effect = Exception("DB Error")
        
        batch = [article1, article2]
        
        result = self.deleter.process_batch(batch)
        
        # Should return empty list on error
        assert result == []
        
        # Should log error
        mock_logger.error.assert_called_once_with("Error deleting embeddings for batch: DB Error")
        
        # Articles should not have their status updated
        from align_data.db.models import PineconeStatus
        assert not hasattr(article1, 'pinecone_status') or article1.pinecone_status != PineconeStatus.absent
        assert not hasattr(article2, 'pinecone_status') or article2.pinecone_status != PineconeStatus.absent

    def test_batch_entries(self):
        """Test batch_entries method."""
        articles = [Mock(spec=Article) for _ in range(15)]
        
        def article_generator():
            for article in articles:
                yield article
        
        batches = list(self.deleter.batch_entries(article_generator()))
        
        # Should have 2 batches: 10 and 5
        assert len(batches) == 2
        assert len(batches[0]) == 10
        assert len(batches[1]) == 5
        
        # Verify all articles are included
        all_batched_articles = []
        for batch in batches:
            all_batched_articles.extend(batch)
        assert len(all_batched_articles) == 15
        assert all_batched_articles == articles

    def test_batch_entries_empty_stream(self):
        """Test batch_entries with empty article stream."""
        def empty_generator():
            return
            yield  # This line is never reached
        
        batches = list(self.deleter.batch_entries(empty_generator()))
        assert len(batches) == 0

    def test_batch_entries_single_batch(self):
        """Test batch_entries with fewer articles than batch size."""
        articles = [Mock(spec=Article) for _ in range(3)]
        
        def article_generator():
            for article in articles:
                yield article
        
        batches = list(self.deleter.batch_entries(article_generator()))
        
        assert len(batches) == 1
        assert len(batches[0]) == 3
        assert batches[0] == articles
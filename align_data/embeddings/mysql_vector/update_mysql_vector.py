import logging
import traceback
from itertools import islice
from typing import Any, Callable, Iterable, List, Tuple, Generator, Iterator

from sqlalchemy.orm import Session
from pydantic import ValidationError

from align_data.embeddings.embedding_utils import get_embeddings
from align_data.db.models import Article, PineconeStatus
from align_data.db.session import make_session
from align_data.embeddings.mysql_vector.mysql_vector_db_handler import MySQLVectorDB
from align_data.embeddings.mysql_vector.mysql_vector_models import MySQLVectorEntry
from align_data.embeddings.text_splitter import split_text

logger = logging.getLogger(__name__)

# Define type aliases for the Callables
LengthFunctionType = Callable[[str], int]
TruncateFunctionType = Callable[[str, int], str]


class MySQLVectorAction:
    """
    Base class for MySQL vector operations that mirrors PineconeAction functionality.
    
    This class provides the foundation for batch processing of articles and their
    embeddings in MySQL, including session management, error handling, and progress logging.
    """
    batch_size = 10

    def __init__(self, mysql_db=None):
        """
        Initialize the MySQL vector action.
        
        Args:
            mysql_db: MySQLVectorDB instance. If None, creates a new instance.
        """
        self.mysql_db = mysql_db or MySQLVectorDB()

    def _articles_by_source(
        self, session: Session, sources: List[str], force_update: bool
    ) -> Iterable[Article]:
        """
        Get articles by source for processing.
        
        This method should be implemented by subclasses to define how articles
        are queried based on sources.
        
        Args:
            session: Database session
            sources: List of source names to filter by
            force_update: Whether to force update regardless of current status
            
        Returns:
            Iterable of Article objects
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    def _articles_by_id(
        self, session: Session, ids: List[str], force_update: bool
    ) -> Iterable[Article]:
        """
        Get articles by ID for processing.
        
        This method should be implemented by subclasses to define how articles
        are queried based on their hash IDs.
        
        Args:
            session: Database session
            ids: List of article hash IDs to filter by
            force_update: Whether to force update regardless of current status
            
        Returns:
            Iterable of Article objects
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    def _update_with_logging(
        self, session: Session, articles_query, log_progress: bool
    ):
        """
        Helper method to handle update logic with consistent logging.
        
        This method processes articles in batches, providing progress updates
        and handling errors gracefully.
        
        Args:
            session: Database session
            articles_query: Query object for articles to process
            log_progress: Whether to log progress updates
        """
        total_articles = articles_query.count()

        if log_progress:
            logger.info("Processing %s items", total_articles)

        total_processed = 0
        for batch in self.batch_entries(articles_query):
            self.save_batch(session, batch)
            total_processed += len(batch)

            if log_progress:
                percentage = (
                    (total_processed / total_articles) * 100
                    if total_articles > 0
                    else 0
                )
                logger.info(
                    "Progress: %.1f%% (%d/%d)",
                    percentage,
                    total_processed,
                    total_articles,
                )

        if log_progress:
            logger.info("Completed processing %s items", total_processed)

    def update(
        self,
        custom_sources: List[str],
        force_update: bool = False,
        log_progress: bool = True,
    ):
        """
        Update the given sources. If no sources are provided, updates all sources.

        Args:
            custom_sources: List of sources to update.
            force_update: Whether to force update regardless of current status.
            log_progress: Whether to log progress updates.
        """
        with make_session() as session:
            articles_to_update = self._articles_by_source(
                session, custom_sources, force_update
            )
            self._update_with_logging(session, articles_to_update, log_progress)

    def update_articles_by_ids(
        self, hash_ids: List[str], force_update: bool = False, log_progress: bool = True
    ):
        """
        Update the MySQL vector entries of specific articles based on their hash_ids.
        
        Args:
            hash_ids: List of article hash IDs to update
            force_update: Whether to force update regardless of current status
            log_progress: Whether to log progress updates
        """
        with make_session() as session:
            articles_to_update = self._articles_by_id(session, hash_ids, force_update)
            self._update_with_logging(session, articles_to_update, log_progress)

    def process_batch(
        self, batch: List[Tuple[Article, MySQLVectorEntry | None]]
    ) -> List[Article]:
        """
        Process a batch of articles and their MySQL vector entries.
        
        This method should be implemented by subclasses to define the specific
        processing logic for each batch.
        
        Args:
            batch: List of tuples containing Article and optional MySQLVectorEntry
            
        Returns:
            List of processed Article objects
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    def save_batch(self, session: Session, batch: List[Any]):
        """
        Save a batch of processed articles with error handling.
        
        This method handles database transactions, committing successful batches
        and rolling back on errors while continuing processing.
        
        Args:
            session: Database session
            batch: List of items to process (format depends on subclass)
        """
        try:
            processed_articles = self.process_batch(batch)
            session.add_all(processed_articles)
            session.commit()

        except Exception as e:
            # Rollback on any kind of error. The next run will redo this batch, 
            # but in the meantime keep trucking
            logger.error("Error processing batch: %s", e)
            traceback.print_exc()
            session.rollback()

    def batch_entries(
        self, article_stream: Generator[Article, None, None], log_progress: bool = True
    ) -> Iterator[List[Article]]:
        """
        Stream articles in batches for processing.
        
        This method takes a generator of articles and yields them in batches
        of the configured batch_size.
        
        Args:
            article_stream: Generator yielding Article objects
            log_progress: Whether to log progress (unused in base implementation)
            
        Yields:
            List[Article]: Batches of articles for processing
        """
        items = iter(article_stream)
        while batch := tuple(islice(items, self.batch_size)):
            yield list(batch)


def get_text_chunks(article: Article) -> List[str]:
    """
    Extract and format text chunks from an article for embedding generation.
    
    This function mirrors the Pinecone implementation's text chunk extraction,
    creating formatted chunks with article metadata.
    
    Args:
        article: Article object to extract text chunks from
        
    Returns:
        List of formatted text chunks ready for embedding generation
    """
    title = article.title.replace("\n", " ")

    authors_lst = [author.strip() for author in article.authors.split(",")]
    authors = get_authors_str(authors_lst)

    signature = f"Title: {title}; Author(s): {authors}."

    text_chunks = split_text(article.text)
    for summary in article.summaries:
        text_chunks += split_text(summary.text)

    return [f'###{signature}###\n"""{text_chunk}"""' for text_chunk in text_chunks]


def get_authors_str(authors_lst: List[str]) -> str:
    """
    Format a list of authors into a readable string.
    
    This function mirrors the Pinecone implementation's author formatting,
    handling various cases and truncating if necessary.
    
    Args:
        authors_lst: List of author names
        
    Returns:
        Formatted author string
    """
    if not authors_lst:
        return "n/a"

    if len(authors_lst) == 1:
        authors_str = authors_lst[0]
    else:
        authors_lst = authors_lst[:4]
        authors_str = f"{', '.join(authors_lst[:-1])} and {authors_lst[-1]}"

    authors_str = authors_str.replace("\n", " ")

    # Truncate if necessary
    if len(authors_str) > 500:
        authors_str = authors_str[:497] + "..."

    return authors_str


class MySQLVectorAdder(MySQLVectorAction):
    """
    MySQL Vector Adder class that mirrors PineconeAdder functionality.
    
    This class handles adding article embeddings to the MySQL vector database,
    processing articles in batches and generating embeddings using the existing
    get_embeddings function.
    """
    batch_size = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _articles_by_source(self, session: Session, sources: List[str], force_update: bool) -> Iterable[Article]:
        """
        Get articles by source for MySQL vector processing.
        
        Args:
            session: Database session
            sources: List of source names to filter by
            force_update: Whether to force update regardless of current status
            
        Returns:
            Iterable of Article objects filtered by source
        """
        from align_data.db.session import get_mysql_vector_articles_by_sources
        return get_mysql_vector_articles_by_sources(session, sources, force_update)

    def _articles_by_id(self, session: Session, ids: List[str], force_update: bool) -> Iterable[Article]:
        """
        Get articles by ID for MySQL vector processing.
        
        Args:
            session: Database session
            ids: List of article hash IDs to filter by
            force_update: Whether to force update regardless of current status
            
        Returns:
            Iterable of Article objects filtered by hash IDs
        """
        from align_data.db.session import get_mysql_vector_articles_by_ids
        return get_mysql_vector_articles_by_ids(session, ids, force_update)

    def process_batch(self, batch: List[Tuple[Article, MySQLVectorEntry | None]]) -> List[Article]:
        """
        Process a batch of articles and their MySQL vector entries.
        
        This method handles the insertion of embeddings into the MySQL database
        and updates the article status accordingly.
        
        Args:
            batch: List of tuples containing Article and optional MySQLVectorEntry
            
        Returns:
            List of processed Article objects with updated status
        """
        logger.info("Processing batch of %s items", len(batch))
        processed_articles = []
        
        for article, mysql_entry in batch:
            try:
                if mysql_entry:
                    self.mysql_db.upsert_entry(mysql_entry)
                    logger.info(f"Successfully processed embeddings for article {article.id}")
                else:
                    logger.warning(f"No MySQL entry generated for article {article.id}")

                # Update article status to indicate successful processing
                article.pinecone_status = PineconeStatus.added
                processed_articles.append(article)
                
            except Exception as e:
                logger.error(f"Error processing article {article.id}: {e}")
                # Don't add to processed_articles so it won't be committed
                # The article will be retried in the next run
                continue
        
        return processed_articles

    def batch_entries(
        self, article_stream: Generator[Article, None, None], log_progress: bool = True
    ) -> Iterator[List[Tuple[Article, MySQLVectorEntry | None]]]:
        """
        Stream articles in batches with their MySQL vector entries.
        
        This method takes a generator of articles and yields them in batches
        with their corresponding MySQLVectorEntry objects.
        
        Args:
            article_stream: Generator yielding Article objects
            log_progress: Whether to log progress (unused in base implementation)
            
        Yields:
            List[Tuple[Article, MySQLVectorEntry | None]]: Batches of articles with their vector entries
        """
        items = iter(article_stream)
        while batch := tuple(islice(items, self.batch_size)):
            yield [(article, self._make_mysql_entry(article)) for article in batch]

    def _make_mysql_entry(self, article: Article) -> MySQLVectorEntry | None:
        """
        Convert Article to MySQLVectorEntry using existing get_embeddings function.
        
        This method mirrors the PineconeAdder._make_pinecone_entry functionality,
        generating embeddings for the article and creating a MySQLVectorEntry.
        
        Args:
            article: Article object to process
            
        Returns:
            MySQLVectorEntry object with embeddings, or None if processing fails
        """
        logger.info(f"Getting embeddings for {article.title}")
        article.comments = ""
        
        try:
            # Get text chunks using the same function as Pinecone
            text_chunks = get_text_chunks(article)
            
            # Generate embeddings using existing function
            embeddings, moderation_results = get_embeddings(text_chunks)
            
            if not embeddings:
                logger.warning(f"No embeddings found for {article.title}")
                logger.warning(f"Moderation results: {moderation_results}")
                logger.info("text: %s", text_chunks)
                return None

            # Handle moderation results
            if moderation_results:
                flagged_text_chunks = [result["text"] for result in moderation_results]
                logger.warning(
                    f"OpenAI moderation flagged text chunks for the following article: {article.id}"
                )
                article.append_comment(
                    f"OpenAI moderation flagged the following text chunks: {flagged_text_chunks}"
                )

            # Create MySQLVectorEntry
            return MySQLVectorEntry(
                article_hash_id=article.id,  # the hash_id of the article
                embeddings=embeddings,
            )
            
        except ValidationError as e:
            logger.warning(f"Validation error for article {article.id}: {e}")
            article.append_comment(
                f"Error encountered while processing this article: {e}"
            )
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing article {article.id}: {e}")
            traceback.print_exc()
            article.append_comment(
                f"Error encountered while processing this article: {e}"
            )
            return None


class MySQLVectorDeleter(MySQLVectorAction):
    """
    MySQL Vector Deleter class that handles deletion of article embeddings.
    
    This class handles removing article embeddings from the MySQL vector database,
    processing articles in batches and updating their status accordingly.
    """
    batch_size = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _articles_by_source(self, session: Session, sources: List[str], force_update: bool) -> Iterable[Article]:
        """
        Get articles by source for MySQL vector deletion.
        
        Args:
            session: Database session
            sources: List of source names to filter by
            force_update: Whether to force deletion regardless of current status
            
        Returns:
            Iterable of Article objects filtered by source for deletion
        """
        from align_data.db.session import get_mysql_vector_to_delete_by_sources
        return get_mysql_vector_to_delete_by_sources(session, sources)

    def _articles_by_id(self, session: Session, ids: List[str], force_update: bool) -> Iterable[Article]:
        """
        Get articles by ID for MySQL vector deletion.
        
        Args:
            session: Database session
            ids: List of article hash IDs to filter by
            force_update: Whether to force deletion regardless of current status
            
        Returns:
            Iterable of Article objects filtered by hash IDs for deletion
        """
        from align_data.db.session import get_mysql_vector_to_delete_by_ids
        return get_mysql_vector_to_delete_by_ids(session, ids)

    def process_batch(self, batch: List[Article]) -> List[Article]:
        """
        Process a batch of articles for embedding deletion.
        
        This method handles the deletion of embeddings from the MySQL database
        and updates the article status accordingly.
        
        Args:
            batch: List of Article objects to delete embeddings for
            
        Returns:
            List of processed Article objects with updated status
        """
        logger.info("Processing deletion batch of %s items", len(batch))
        processed_articles = []
        
        # Extract article hash IDs for batch deletion
        article_hash_ids = [article.id for article in batch]
        
        try:
            # Delete embeddings for all articles in the batch
            self.mysql_db.delete_entries(article_hash_ids)
            
            # Update status for all articles in the batch
            for article in batch:
                # Update article status to indicate successful deletion
                article.pinecone_status = PineconeStatus.absent
                processed_articles.append(article)
                logger.info(f"Successfully deleted embeddings for article {article.id}")
                
        except Exception as e:
            logger.error(f"Error deleting embeddings for batch: {e}")
            # Don't add any articles to processed_articles so none will be committed
            # The articles will be retried in the next run
            return []
        
        return processed_articles

    def batch_entries(
        self, article_stream: Generator[Article, None, None], log_progress: bool = True
    ) -> Iterator[List[Article]]:
        """
        Stream articles in batches for deletion processing.
        
        This method takes a generator of articles and yields them in batches
        for deletion processing.
        
        Args:
            article_stream: Generator yielding Article objects
            log_progress: Whether to log progress (unused in base implementation)
            
        Yields:
            List[Article]: Batches of articles for deletion processing
        """
        items = iter(article_stream)
        while batch := tuple(islice(items, self.batch_size)):
            yield list(batch)
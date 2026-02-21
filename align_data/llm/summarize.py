import json
import logging
import traceback
from itertools import islice
from typing import Iterator, List

from sqlalchemy.orm import Session

from align_data.db.models import Article
from align_data.db.session import make_session
from align_data.llm.provider import LLMProvider, create_llm_provider

logger = logging.getLogger(__name__)


class ArticleSummarizer:
    batch_size = 10

    def __init__(self):
        self.provider: LLMProvider = create_llm_provider()

    def update(
        self,
        sources: List[str],
        force_update: bool = False,
        log_progress: bool = True,
    ):
        """Summarize articles from the given sources."""
        with make_session() as session:
            query = self._build_query(session, sources, force_update)
            self._process_articles(session, query, log_progress)

    def update_articles_by_ids(
        self,
        hash_ids: List[str],
        force_update: bool = False,
        log_progress: bool = True,
    ):
        """Summarize specific articles by their hash IDs."""
        with make_session() as session:
            query = session.query(Article).filter(
                Article.is_valid,
                Article.id.in_(hash_ids),
            )
            if not force_update:
                query = query.filter(Article.summary == None)
            self._process_articles(session, query, log_progress)

    def _build_query(self, session: Session, sources: List[str], force_update: bool):
        query = session.query(Article).filter(Article.is_valid)
        if sources:
            query = query.filter(Article.source.in_(sources))
        if not force_update:
            query = query.filter(Article.summary == None)
        return query

    def _process_articles(self, session: Session, query, log_progress: bool):
        total = query.count()
        if log_progress:
            logger.info("Processing %s articles for summarization", total)

        processed = 0
        for batch in self._batch_entries(query):
            self._process_batch(session, batch)
            processed += len(batch)
            if log_progress:
                pct = (processed / total * 100) if total > 0 else 0
                logger.info(
                    "Progress: %.1f%% (%d/%d)", pct, processed, total
                )

        if log_progress:
            logger.info("Completed summarization of %s articles", processed)

    def _batch_entries(self, query) -> Iterator[List[Article]]:
        items = iter(query)
        while batch := list(islice(items, self.batch_size)):
            yield batch

    def _process_batch(self, session: Session, batch: List[Article]):
        try:
            for article in batch:
                try:
                    analysis = self.provider.analyze_article(
                        title=article.title or "",
                        text=article.text or "",
                        source=article.source or "",
                    )
                    article.summary = analysis.summary
                    article.key_points = json.dumps(analysis.key_points)
                    article.implication = analysis.implication
                    article.category = analysis.category
                    session.add(article)
                except Exception as e:
                    logger.error(
                        "Error summarizing article %s: %s", article.id, e
                    )
                    traceback.print_exc()
            session.commit()
        except Exception as e:
            logger.error("Error committing batch: %s", e)
            traceback.print_exc()
            session.rollback()

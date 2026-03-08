import json
import logging
import traceback
from itertools import islice
from typing import Iterator, List

from sqlalchemy import update
from sqlalchemy.orm import Session

from align_data.db.models import Article
from align_data.db.session import make_session
from align_data.llm.provider import LLMProvider, TokenUsage, create_llm_provider

logger = logging.getLogger(__name__)


MIN_TEXT_LENGTH = 200

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

        usage = TokenUsage()
        processed = 0
        for batch in self._batch_entries(query):
            self._process_batch(session, batch, usage)
            processed += len(batch)
            if log_progress:
                pct = (processed / total * 100) if total > 0 else 0
                logger.info(
                    "Progress: %.1f%% (%d/%d)", pct, processed, total
                )

        if log_progress:
            logger.info("Completed summarization of %s articles", processed)

        if usage.total_tokens > 0:
            logger.info(
                "Total token usage for %s: %d prompt, %d completion, %d total",
                usage.model, usage.prompt_tokens, usage.completion_tokens, usage.total_tokens,
            )

    def _batch_entries(self, query) -> Iterator[List[Article]]:
        items = iter(query)
        while batch := list(islice(items, self.batch_size)):
            yield batch

    def _process_batch(self, session: Session, batch: List[Article], usage: TokenUsage):
        try:
            for article in batch:
                text = article.text or ""
                if len(text.strip()) < MIN_TEXT_LENGTH:
                    logger.info(
                        "Skipping article %s (%s): text too short (%d chars)",
                        article.id, article.title, len(text.strip()),
                    )
                    continue
                try:
                    result = self.provider.analyze_article(
                        title=article.title or "",
                        text=text,
                        source=article.source or "",
                    )
                    usage.prompt_tokens += result.usage.prompt_tokens
                    usage.completion_tokens += result.usage.completion_tokens
                    usage.total_tokens += result.usage.total_tokens
                    usage.model = result.usage.model
                    # Use a targeted UPDATE to avoid flushing unrelated fields
                    # (the JSON 'meta' column can cause serialization errors
                    # with mysql-connector-python's C extension).
                    session.execute(
                        update(Article)
                        .where(Article._id == article._id)
                        .values(
                            summary=result.analysis.summary,
                            key_points=json.dumps(result.analysis.key_points),
                            implication=result.analysis.implication,
                            category=result.analysis.category,
                        )
                    )
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

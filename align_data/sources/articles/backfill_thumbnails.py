import logging
import time
import traceback
from itertools import islice
from typing import Iterator, List

from sqlalchemy import update
from sqlalchemy.orm import Session

from align_data.db.models import Article
from align_data.db.session import make_session
from align_data.common.thumbnails import fetch_thumbnail_url, youtube_thumbnail_url

logger = logging.getLogger(__name__)


class ThumbnailBackfiller:
    batch_size = 50
    request_delay = 1.0  # seconds between HTTP requests

    def update(
        self,
        sources: List[str],
        force_update: bool = False,
        log_progress: bool = True,
    ):
        """Backfill thumbnails for articles from the given sources."""
        with make_session() as session:
            query = self._build_query(session, sources, force_update)
            self._process_articles(session, query, log_progress)

    def update_articles_by_ids(
        self,
        hash_ids: List[str],
        force_update: bool = False,
        log_progress: bool = True,
    ):
        """Backfill thumbnails for specific articles by their hash IDs."""
        with make_session() as session:
            query = session.query(Article).filter(
                Article.is_valid,
                Article.id.in_(hash_ids),
            )
            if not force_update:
                query = query.filter(Article.thumbnail_url == None)
            self._process_articles(session, query, log_progress)

    def _build_query(self, session: Session, sources: List[str], force_update: bool):
        query = session.query(Article).filter(Article.is_valid)
        if sources:
            query = query.filter(Article.source.in_(sources))
        if not force_update:
            query = query.filter(Article.thumbnail_url == None)
        return query

    def _process_articles(self, session: Session, query, log_progress: bool):
        total = query.count()
        if log_progress:
            logger.info("Processing %s articles for thumbnail backfill", total)

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
            logger.info("Completed thumbnail backfill of %s articles", processed)

    def _batch_entries(self, query) -> Iterator[List[Article]]:
        items = iter(query)
        while batch := list(islice(items, self.batch_size)):
            yield batch

    def _process_batch(self, session: Session, batch: List[Article]):
        try:
            for article in batch:
                try:
                    thumbnail = self._extract_thumbnail(article)
                    if thumbnail:
                        session.execute(
                            update(Article)
                            .where(Article._id == article._id)
                            .values(thumbnail_url=thumbnail)
                        )
                except Exception as e:
                    logger.error(
                        "Error extracting thumbnail for article %s: %s",
                        article.id, e,
                    )
                    traceback.print_exc()
            session.commit()
        except Exception as e:
            logger.error("Error committing batch: %s", e)
            traceback.print_exc()
            session.rollback()

    def _extract_thumbnail(self, article: Article):
        url = article.url
        if not url:
            return None

        is_youtube = youtube_thumbnail_url(url) is not None
        thumbnail = fetch_thumbnail_url(url)

        # Rate-limit only when an HTTP request was made
        if not is_youtube:
            time.sleep(self.request_delay)

        return thumbnail

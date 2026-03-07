from unittest.mock import MagicMock, patch

import pytest

from align_data.sources.articles.backfill_thumbnails import ThumbnailBackfiller

# Mock fetch where it's imported by thumbnails.py
FETCH_PATH = "align_data.common.thumbnails.fetch"


class TestExtractThumbnail:
    def setup_method(self):
        self.backfiller = ThumbnailBackfiller()

    def _make_article(self, url=None):
        article = MagicMock()
        article.url = url
        return article

    def test_no_url_returns_none(self):
        article = self._make_article(url=None)
        assert self.backfiller._extract_thumbnail(article) is None

    def test_empty_url_returns_none(self):
        article = self._make_article(url="")
        assert self.backfiller._extract_thumbnail(article) is None

    def test_youtube_url_returns_deterministic_thumbnail(self):
        article = self._make_article(url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        result = self.backfiller._extract_thumbnail(article)
        assert result == "https://img.youtube.com/vi/dQw4w9WgXcQ/mqdefault.jpg"

    @patch(FETCH_PATH)
    def test_youtube_url_no_http_request(self, mock_fetch):
        article = self._make_article(url="https://youtu.be/dQw4w9WgXcQ")
        result = self.backfiller._extract_thumbnail(article)
        mock_fetch.assert_not_called()
        assert result == "https://img.youtube.com/vi/dQw4w9WgXcQ/mqdefault.jpg"

    @patch("align_data.sources.articles.backfill_thumbnails.time.sleep")
    @patch(FETCH_PATH)
    def test_html_with_og_image(self, mock_fetch, mock_sleep):
        resp = MagicMock()
        resp.ok = True
        resp.headers = {"Content-Type": "text/html; charset=utf-8"}
        resp.content = (
            b'<html><head><meta property="og:image" content="https://example.com/thumb.jpg">'
            b"</head><body></body></html>"
        )
        mock_fetch.return_value = resp

        article = self._make_article(url="https://example.com/article")
        result = self.backfiller._extract_thumbnail(article)
        assert result == "https://example.com/thumb.jpg"

    @patch("align_data.sources.articles.backfill_thumbnails.time.sleep")
    @patch(FETCH_PATH)
    def test_html_without_og_image_returns_none(self, mock_fetch, mock_sleep):
        resp = MagicMock()
        resp.ok = True
        resp.headers = {"Content-Type": "text/html; charset=utf-8"}
        resp.content = b"<html><head></head><body>Hello</body></html>"
        mock_fetch.return_value = resp

        article = self._make_article(url="https://example.com/article")
        assert self.backfiller._extract_thumbnail(article) is None

    @patch("align_data.sources.articles.backfill_thumbnails.time.sleep")
    @patch(FETCH_PATH)
    def test_non_html_content_type_returns_none(self, mock_fetch, mock_sleep):
        resp = MagicMock()
        resp.ok = True
        resp.headers = {"Content-Type": "application/pdf"}
        mock_fetch.return_value = resp

        article = self._make_article(url="https://example.com/paper.pdf")
        assert self.backfiller._extract_thumbnail(article) is None

    @patch("align_data.sources.articles.backfill_thumbnails.time.sleep")
    @patch(FETCH_PATH)
    def test_failed_http_request_returns_none(self, mock_fetch, mock_sleep):
        resp = MagicMock()
        resp.ok = False
        mock_fetch.return_value = resp

        article = self._make_article(url="https://example.com/article")
        assert self.backfiller._extract_thumbnail(article) is None

    @patch("align_data.sources.articles.backfill_thumbnails.time.sleep")
    @patch(FETCH_PATH)
    def test_fetch_exception_returns_none(self, mock_fetch, mock_sleep):
        mock_fetch.side_effect = Exception("Connection error")

        article = self._make_article(url="https://example.com/article")
        assert self.backfiller._extract_thumbnail(article) is None

    @patch("align_data.sources.articles.backfill_thumbnails.time.sleep")
    @patch(FETCH_PATH)
    def test_fetch_returns_none(self, mock_fetch, mock_sleep):
        mock_fetch.return_value = None

        article = self._make_article(url="https://example.com/article")
        assert self.backfiller._extract_thumbnail(article) is None

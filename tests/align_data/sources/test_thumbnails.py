import pytest
from bs4 import BeautifulSoup

from align_data.common.thumbnails import (
    extract_thumbnail_url,
    youtube_thumbnail_url,
    extract_thumbnail_from_html,
    resolve_thumbnail_url,
)


class TestYoutubeThumbnailUrl:
    def test_watch_url(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert youtube_thumbnail_url(url) == "https://img.youtube.com/vi/dQw4w9WgXcQ/mqdefault.jpg"

    def test_short_url(self):
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert youtube_thumbnail_url(url) == "https://img.youtube.com/vi/dQw4w9WgXcQ/mqdefault.jpg"

    def test_embed_url(self):
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        assert youtube_thumbnail_url(url) == "https://img.youtube.com/vi/dQw4w9WgXcQ/mqdefault.jpg"

    def test_watch_url_with_extra_params(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120"
        assert youtube_thumbnail_url(url) == "https://img.youtube.com/vi/dQw4w9WgXcQ/mqdefault.jpg"

    def test_non_youtube_url(self):
        assert youtube_thumbnail_url("https://example.com/article") is None

    def test_youtube_channel_url(self):
        assert youtube_thumbnail_url("https://www.youtube.com/channel/UC123") is None


class TestResolveThumbailUrl:
    def test_absolute_url(self):
        result = resolve_thumbnail_url("https://example.com", "https://cdn.example.com/img.jpg")
        assert result == "https://cdn.example.com/img.jpg"

    def test_relative_url(self):
        result = resolve_thumbnail_url("https://example.com/blog/post", "/images/thumb.jpg")
        assert result == "https://example.com/images/thumb.jpg"

    def test_protocol_relative_url(self):
        result = resolve_thumbnail_url("https://example.com", "//cdn.example.com/img.jpg")
        assert result == "https://cdn.example.com/img.jpg"

    def test_overlong_url_returns_none(self):
        long_url = "https://example.com/" + "a" * 2048
        assert resolve_thumbnail_url("https://example.com", long_url) is None

    def test_empty_url_returns_none(self):
        assert resolve_thumbnail_url("https://example.com", "") is None

    def test_whitespace_url_returns_none(self):
        assert resolve_thumbnail_url("https://example.com", "   ") is None

    def test_invalid_scheme_returns_none(self):
        assert resolve_thumbnail_url("https://example.com", "ftp://example.com/img.jpg") is None

    def test_http_url_allowed(self):
        result = resolve_thumbnail_url("https://example.com", "http://cdn.example.com/img.jpg")
        assert result == "http://cdn.example.com/img.jpg"


class TestExtractThumbnailFromHtml:
    def _make_soup(self, meta_tag):
        return BeautifulSoup(f"<html><head>{meta_tag}</head><body></body></html>", "html.parser")

    def test_og_image(self):
        soup = self._make_soup('<meta property="og:image" content="https://example.com/og.jpg">')
        assert extract_thumbnail_from_html("https://example.com", soup) == "https://example.com/og.jpg"

    def test_twitter_image(self):
        soup = self._make_soup('<meta name="twitter:image" content="https://example.com/tw.jpg">')
        assert extract_thumbnail_from_html("https://example.com", soup) == "https://example.com/tw.jpg"

    def test_twitter_image_src(self):
        soup = self._make_soup('<meta name="twitter:image:src" content="https://example.com/tw2.jpg">')
        assert extract_thumbnail_from_html("https://example.com", soup) == "https://example.com/tw2.jpg"

    def test_og_image_secure_url(self):
        soup = self._make_soup('<meta property="og:image:secure_url" content="https://example.com/secure.jpg">')
        assert extract_thumbnail_from_html("https://example.com", soup) == "https://example.com/secure.jpg"

    def test_og_image_takes_priority_over_twitter(self):
        soup = self._make_soup(
            '<meta property="og:image" content="https://example.com/og.jpg">'
            '<meta name="twitter:image" content="https://example.com/tw.jpg">'
        )
        assert extract_thumbnail_from_html("https://example.com", soup) == "https://example.com/og.jpg"

    def test_missing_tags_returns_none(self):
        soup = self._make_soup("")
        assert extract_thumbnail_from_html("https://example.com", soup) is None

    def test_empty_content_returns_none(self):
        soup = self._make_soup('<meta property="og:image" content="">')
        assert extract_thumbnail_from_html("https://example.com", soup) is None

    def test_relative_og_image(self):
        soup = self._make_soup('<meta property="og:image" content="/images/og.jpg">')
        assert extract_thumbnail_from_html("https://example.com/blog/post", soup) == "https://example.com/images/og.jpg"


class TestExtractThumbnailUrl:
    def test_youtube_url_without_soup(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_thumbnail_url(url) == "https://img.youtube.com/vi/dQw4w9WgXcQ/mqdefault.jpg"

    def test_youtube_url_with_soup(self):
        """YouTube pattern should take priority even when soup has og:image."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        soup = BeautifulSoup(
            '<html><head><meta property="og:image" content="https://other.com/img.jpg"></head></html>',
            "html.parser",
        )
        assert extract_thumbnail_url(url, soup) == "https://img.youtube.com/vi/dQw4w9WgXcQ/mqdefault.jpg"

    def test_non_youtube_with_soup(self):
        url = "https://blog.example.com/post"
        soup = BeautifulSoup(
            '<html><head><meta property="og:image" content="https://blog.example.com/thumb.jpg"></head></html>',
            "html.parser",
        )
        assert extract_thumbnail_url(url, soup) == "https://blog.example.com/thumb.jpg"

    def test_non_youtube_without_soup(self):
        assert extract_thumbnail_url("https://blog.example.com/post") is None

    def test_non_youtube_with_empty_soup(self):
        soup = BeautifulSoup("<html><head></head></html>", "html.parser")
        assert extract_thumbnail_url("https://blog.example.com/post", soup) is None

import logging
import re
from urllib.parse import urljoin, urlparse
from typing import Optional

from bs4 import BeautifulSoup

from align_data.common.http import fetch

logger = logging.getLogger(__name__)


MAX_URL_LENGTH = 2048

# Matches youtube.com/watch?v=ID, youtu.be/ID, youtube.com/embed/ID
_YOUTUBE_PATTERNS = [
    re.compile(r"(?:youtube\.com/watch\?.*v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})"),
]


def youtube_thumbnail_url(url: str) -> Optional[str]:
    """Derive a thumbnail URL from a YouTube video URL."""
    for pattern in _YOUTUBE_PATTERNS:
        match = pattern.search(url)
        if match:
            video_id = match.group(1)
            return f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"
    return None


def resolve_thumbnail_url(base_url: str, image_url: str) -> Optional[str]:
    """Resolve a potentially relative image URL and validate it."""
    if not image_url or not image_url.strip():
        return None

    image_url = image_url.strip()

    # Handle protocol-relative URLs
    if image_url.startswith("//"):
        image_url = "https:" + image_url

    # Resolve relative URLs
    if not image_url.startswith(("http://", "https://")):
        image_url = urljoin(base_url, image_url)

    # Validate scheme
    parsed = urlparse(image_url)
    if parsed.scheme not in ("http", "https"):
        return None

    # Enforce length limit
    if len(image_url) > MAX_URL_LENGTH:
        return None

    return image_url


def extract_thumbnail_from_html(url: str, soup: BeautifulSoup) -> Optional[str]:
    """Extract a thumbnail URL from HTML meta tags."""
    # Priority order for meta tags
    selectors = [
        ("property", "og:image"),
        ("name", "twitter:image"),
        ("name", "twitter:image:src"),
        ("property", "og:image:secure_url"),
    ]

    for attr, value in selectors:
        tag = soup.find("meta", attrs={attr: value})
        if tag and tag.get("content"):
            resolved = resolve_thumbnail_url(url, tag["content"])
            if resolved:
                return resolved

    return None


def extract_thumbnail_url(url: str, soup: Optional[BeautifulSoup] = None) -> Optional[str]:
    """Extract a thumbnail URL for an article.

    Tries YouTube URL pattern matching first, then falls back to HTML meta tags.
    """
    # Try YouTube pattern first (no HTML needed)
    yt_thumb = youtube_thumbnail_url(url)
    if yt_thumb:
        return yt_thumb

    # Try HTML meta tags if soup is available
    if soup is not None:
        return extract_thumbnail_from_html(url, soup)

    return None


def fetch_thumbnail_url(url: str) -> Optional[str]:
    """Extract a thumbnail URL, fetching the page if needed.

    Tries YouTube URL pattern matching first (no HTTP), then fetches the page
    to extract og:image / twitter:image meta tags.
    """
    yt_thumb = youtube_thumbnail_url(url)
    if yt_thumb:
        return yt_thumb

    try:
        resp = fetch(url)
    except Exception:
        logger.debug("Failed to fetch %s for thumbnail extraction", url)
        return None

    if not resp or not resp.ok:
        return None

    content_type = resp.headers.get("Content-Type", "")
    if "text/html" not in content_type:
        return None

    soup = BeautifulSoup(resp.content, "html.parser")
    return extract_thumbnail_from_html(url, soup)

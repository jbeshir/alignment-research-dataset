import logging
from typing import Optional, Dict, Any, List

import requests
from bs4 import BeautifulSoup, Tag
from markdownify import MarkdownConverter

from align_data.common.http import fetch, DEFAULT_HEADERS
from align_data.common.thumbnails import extract_thumbnail_url

logger = logging.getLogger(__name__)


def fetch_element(url: str, selector: str, headers: Dict[str, str] = DEFAULT_HEADERS) -> Tag | None:
    """Fetch the first HTML element that matches the given CSS `selector` on the page found at `url`."""
    try:
        resp = fetch(url, headers=headers)
    except requests.exceptions.ConnectionError:
        logger.error("Could not connect to %s", url)
        return None

    soup = BeautifulSoup(resp.content, "html.parser")
    return soup.select_one(selector)


def element_extractor(selector: str, remove: Optional[List[str]] = None):
    """Returns a function that will extract the first element that matches the given CSS selector.

    :params str selector: a CSS selector to run on the HTML of the page provided as the parameter of the function
    :param List[str] remove: An optional list of selectors to be removed from the resulting HTML. Useful for removing footers etc.
    :returns: A function that expects to get an URL, and which will then return the contents of the selected HTML element as markdown.
    """
    remove = remove or []

    def getter(url: str) -> Dict[str, Any]:
        try:
            resp = fetch(url)
        except requests.exceptions.ConnectionError:
            logger.error("Could not connect to %s", url)
            return {}

        soup = BeautifulSoup(resp.content, "html.parser")
        elem = soup.select_one(selector)
        if not elem:
            return {}

        for sel in remove:
            for e in elem.select(sel):
                e.extract()

        result: Dict[str, Any] = {
            "text": MarkdownConverter().convert_soup(elem).strip(),
            "source_url": url,
            "source_type": "html",
        }

        thumbnail = extract_thumbnail_url(url, soup)
        if thumbnail:
            result["thumbnail_url"] = thumbnail

        return result

    return getter

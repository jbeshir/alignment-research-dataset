import time
import logging
from typing import Dict, Literal

import requests

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/113.0",
}


def with_retry(times=3, exceptions=requests.exceptions.RequestException):
    """A decorator that will retry the wrapped function up to `times` times in case of google sheets errors."""

    def wrapper(f):
        def retrier(*args, **kwargs):
            for i in range(times):
                try:
                    return f(*args, **kwargs)
                except exceptions as e:
                    logger.error(f"{e} - retrying up to {times - i} times")
                    # Do a logarithmic backoff
                    time.sleep((i + 1) ** 2)
            raise ValueError(f"Gave up after {times} tries")

        return retrier

    return wrapper


def fetch(
    url: str,
    method: Literal["get", "post", "put", "delete", "patch", "options", "head"] = "get",
    headers: Dict[str, str] = DEFAULT_HEADERS
) -> requests.Response:
    """Fetch the given `url`.

    This function is to have a single place to manage headers etc.
    """
    try:
        return getattr(requests, method)(url, allow_redirects=True, headers=headers)
    except requests.exceptions.TooManyRedirects as e:
        return e.response

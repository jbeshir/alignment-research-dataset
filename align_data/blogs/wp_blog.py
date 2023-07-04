from datetime import datetime, timezone
from calendar import c
from dataclasses import dataclass, field
import logging
import feedparser

from align_data.common import utils
from align_data.common.alignment_dataset import AlignmentDataset, DataEntry

from typing import List

logger = logging.getLogger(__name__)

@dataclass
class WordpressBlog(AlignmentDataset):
    url: str
    strip: List = field(default_factory=lambda: [])
    summary_key = 'summary'
    done_key = 'paged_url'

    def setup(self):
        """
        url: URL of the blog
        strip: list of regexes to strip from the HTML
        """
        super().setup()
        self.feed_url = self.url + "/feed"
        self.cleaner = utils.HtmlCleaner(self.strip)
        self.name = utils.url_to_filename(self.url)

    def get_item_key(self, item):
        return item
    
    @property
    def items_list(self):
        logger.info(f"Fetching entries from {self.feed_url}")

        pages = []
        page_number = 0
        last_title = None
        while True:
            paged_url = f"{self.feed_url}?paged={page_number + 1}"
            logging.info(f"Fetching {paged_url}")

            feed = feedparser.parse(paged_url)
            if (("feed" not in feed) or ("title" not in feed["feed"]) or (feed["feed"]["title"] == last_title)):
                break
            last_title = feed["feed"]["title"]

            pages.extend({**entry, 'paged_url': paged_url} for entry in feed['entries'])
            page_number += 1

        logger.info(f'Got {len(pages)} pages')

        return pages

    @staticmethod
    def _get_published_date(item):
        date_published = item.get('published')
        if not date_published:
            return ''
        dt = datetime.strptime(date_published, '%a, %d %b %Y %H:%M:%S %z').astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    def fetch_entries(self):
        for entry in self.unprocessed_items():
            content_text = self.cleaner.clean(entry["content"][0]["value"])
            text = entry["title"] + "\n\n" + content_text
            
            new_entry = DataEntry({
                "text": text,
                "url": entry['link'],
                "title": text.split("\n")[0],
                "source": self.name,
                "source_type": "blog",
                "date_published": self._get_published_date(entry),
                "paged_url": entry['paged_url'],
                "authors": [e['name'] for e in entry.get('authors', [])],
            })
            new_entry.add_id()

            yield new_entry

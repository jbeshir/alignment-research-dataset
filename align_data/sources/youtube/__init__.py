from align_data.common.alignment_dataset import MultiDataset
from align_data.sources.youtube.youtube import (
    YouTubeChannelDataset,
    YouTubePlaylistDataset,
)

YOUTUBE_DATASETS = []


YOUTUBE_REGISTRY = [
    MultiDataset(name="youtube", datasets=YOUTUBE_DATASETS),
]

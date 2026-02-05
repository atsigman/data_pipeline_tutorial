from dataclasses import dataclass
from typing import List


@dataclass
class TextCondition:
    artist: str
    album_title: str
    track_title: str
    genres: List[str]
    description: str
    tempo: int

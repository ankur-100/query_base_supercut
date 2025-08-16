from typing import List, Dict
from urllib.parse import urlparse
from youtube_transcript_api import YouTubeTranscriptApi
from pydantic import BaseModel
from prompt import GET_RELEVANT_CHUNKS_PROMPT

import litellm
import re

class Chunk(BaseModel):
    id: int
    text: str
    start: float
    duration: float

    def __repr__(self):
        return f"""
======== CHUNK INFO ======== 
Id: {self.id}
Text: {self.text}
Start: {self.start}
Duration: {self.duration}
"""
    
    def __str__(self):
        return f"""
======== CHUNK INFO ======== 
Id: {self.id}
Text: {self.text}
Start: {self.start}
Duration: {self.duration}
"""


def parse_video_id(url_or_id: str) -> str:

    s = url_or_id.strip()

    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s
    
    query = urlparse(s).query
    match = re.search(r"[A-Za-z0-9_-]{11}", query)
    if match is not None:
        return match.group(0)
    
    raise ValueError(f'not a valid url or id: {url_or_id}')


def get_transcript(url_or_id: str) -> List[Chunk]:

    vid_id = parse_video_id(url_or_id)
    ytt_api = YouTubeTranscriptApi()
    fetched_transcripts = ytt_api.fetch(vid_id)
    fetched_transcripts = fetched_transcripts.to_raw_data()
    fetched_transcripts = [
        Chunk(**{"id": i, **ft}) for i, ft in enumerate(fetched_transcripts)
    ]
    return fetched_transcripts


def get_relevant_chunks(url_or_id: str, query: str) -> List[Chunk]:
    
    class RelevantChunks(BaseModel):
        ids: List[int]

    chunks = get_transcript(url_or_id)
    completion = litellm.completion(
        model="gemini/gemini-2.0-flash",
        messages=[
            {"role": "system", "content": GET_RELEVANT_CHUNKS_PROMPT},
            {"role": "user", "content": f"Chunks: {chunks}\n\nQuery: {query}"}
        ],
        response_format=RelevantChunks,
        temperature=0,
        max_tokens=512
    )
    ids = RelevantChunks.model_validate_json(completion.choices[0].message.content).ids
    return [chunks[_id] for _id in ids]


if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=XMGvGvp2a6M"
    query = "negatives of the movie"
    relevant_chunks = get_relevant_chunks(url, query)
    for chunk in relevant_chunks:
        print(chunk)
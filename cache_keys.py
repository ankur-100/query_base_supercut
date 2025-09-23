"""
Centralized key/field naming for the Redis hash-per-video design.

Each video uses exactly one Redis HASH key:
    {NAMESPACE}:{VERSION}:video:{video_id}

Fields we store in that hash:
    - audio    : raw audio bytes (container from yt-dlp; usually not gzipped)
    - transcript : Deepgram transcript JSON (gzipped bytes)
    - segments : list[dict] sentence segments JSON (gzipped bytes)
    - emb      : NumPy ndarray (.npy bytes, gzipped)
    - faiss    : serialized FAISS index (gzipped)
    - fulltxt  : full transcript as plain text (utf-8, gzipped)
    - meta     : small JSON metadata (gzipped)
"""

from __future__ import annotations
import hashlib
import json
from typing import Any

# Bump VERSION when you change preprocessing/model choices to invalidate old caches.
NAMESPACE = "videoproc"
VERSION   = "v1"

# --- Hash key builder ---
def video_hash_key(video_id: str) -> str:
    """Return the Redis hash key for a given video_id."""
    return f"{NAMESPACE}:{VERSION}:video:{video_id}"

# --- Standard field names (use these everywhere) ---
FIELD_AUDIO      = "audio"
FIELD_TRANSCRIPT = "transcript"
FIELD_SEGMENTS   = "segments"
FIELD_EMB        = "emb"
FIELD_FAISS      = "faiss"
FIELD_FULLTXT    = "fulltxt"
FIELD_META       = "meta"

# Optional: small, deterministic hash for content-addressing / fingerprints
def stable_hash(obj: Any) -> str:
    """Deterministic 16-hex hash for any JSON-serializable object."""
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

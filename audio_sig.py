import hashlib
import subprocess
import shutil
from pathlib import Path

def compute_audio_hash_pcm(audio_path: str, rate: int = 16000, channels: int = 1) -> str:
    """
    Compute a content-stable SHA-256 by hashing a canonical PCM stream:
    mono, 16 kHz, 16-bit signed little-endian (s16le), no metadata.
    Requires ffmpeg in PATH.
    """
    if not shutil.which("ffmpeg"):
        # Fallback: hash the file bytes directly (less stable across runs)
        return compute_file_sha256(audio_path)

    cmd = [
        "ffmpeg", "-v", "error", "-i", audio_path,
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ac", str(channels), "-ar", str(rate),
        "-map_metadata", "-1",  # drop metadata
        "-"  # write raw PCM to stdout
    ]
    h = hashlib.sha256()
    # Stream PCM frames and hash incrementally
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        for chunk in iter(lambda: p.stdout.read(1 << 20), b""):
            h.update(chunk)
        p.stdout.close()
        p.wait()

        if p.returncode != 0:
            # If ffmpeg failed for any reason, fallback to file hash
            return compute_file_sha256(audio_path)

    return h.hexdigest()

def compute_file_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    """
    Simple file hash (container-dependent). Use only as fallback.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

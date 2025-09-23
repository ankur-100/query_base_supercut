import torch
import mimetypes
import os, subprocess
import yt_dlp
import whisper
import whisperx
from deepgram import DeepgramClient, PrerecordedOptions
from sentence_transformers import SentenceTransformer, util
import faiss                  # FAISS
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import re, json, ast
from typing import List, Dict, Any, Tuple
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
import nltk
from gtts import gTTS
from pyannote.audio import Pipeline
from pyannote.core import Segment
from dotenv import load_dotenv
import logging            # logging
import os, time, gzip, hashlib, random
from typing import Optional, Any, Dict
import redis
import zlib
import orjson
import io
import uuid

# Load environment variables
load_dotenv()



# Download the sentence tokenizer model from NLTK
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')



# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
JOB_DATA_FOLDER = 'job_data'
TRANSCRIPTION_MODEL = "base.en"
EMBEDDING_MODEL = 'BAAI/bge-base-en-v1.5'
# alternative at compute-avail: 'thenlper/gte-large'
NARRATIVE_MODEL = "google/gemma-3-270m-it"
QA_MODEL = "google/gemma-3-270m-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {DEVICE}")
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
NUM_RE  = r'-?\d+(?:\.\d+)?'
SENT_END_RE = re.compile(r'([.!?…])(\s+|$)')  # sentence boundary
NEED_MP3           = False      # True → re-encode to mp3 (slower). False → keep native m4a/webm and skip re-encode (fastest)
FAST_MP3_Q         = "4"        # mp3 VBR quality if NEED_MP3=True (2=high quality/slower, 4=faster)
HTTP_CHUNK         = 1 * 1024 * 1024   # 1 MiB HTTP chunk size to reduce per-request overhead
FAST_CONC_FRAGS    = 6          # fast attempt: parallel fragment downloads
SAFE_CONC_FRAGS    = 1          # safe fallback: single fragment (avoids 403 storms)
USE_ANDROID_FIRST  = True       # try Android player client first (often less throttled)
TITLE_FALLBACK     = "video"    # fallback title when missing
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DG = DeepgramClient(DG_API_KEY)

# --- Initialize YouTube Transcript API ---

ytt = YouTubeTranscriptApi()

# --- Helper Functions ---

def update_job_status(job_id, jobs, status, progress, message): 
    """Updates the shared job status dictionary."""
    if job_id in jobs:
        jobs[job_id] = {
            'status': status,
            'progress': progress,
            'message': message,
            'result': jobs[job_id].get('result') 
        }

def get_video_id(url): 
    """Extracts the YouTube video ID from a URL."""
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return None

def generate_gemma_response(messages, model_name): 
    """Generates a response from a Gemma model using the chat template."""
    if not HF_TOKEN:
        raise ValueError("HUGGING_FACE_TOKEN environment variable not set. Please create a .env file.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        token=HF_TOKEN,
        torch_dtype=torch.float32, 
        device_map="cpu"
    )
    
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.eos_token_id
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(input_ids=inputs, max_new_tokens=1500, use_cache=True)
    
    response_ids = outputs[0, inputs.shape[1]:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    return response_text

def to_segments(iterable):
    """Normalize transcript entries (object or dict) -> list[{'text','start','end'}]."""
    out = []
    for item in iterable:
        # object style (FetchedTranscriptSnippet)
        if hasattr(item, "text") and hasattr(item, "start") and hasattr(item, "duration"):
            start = float(item.start)
            dur = float(item.duration)
            out.append({"text": item.text, "start": start, "end": start + dur})
        else:
            # dict style (official PyPI API)
            start = float(item["start"])
            dur = float(item["duration"])
            out.append({"text": item["text"], "start": start, "end": start + dur})
    return out

def extract_json_array(response_text: str): 
    """Robustly extracts a JSON array from a model's text response."""
    s = response_text.strip()
    if (s.startswith(("'", '"')) and s.endswith(("'", '"'))) and ("\\n" in s or "\\\"" in s):
        try:
            s = ast.literal_eval(s)
        except Exception:
            s = s[1:-1]
    
    s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)

    start = s.find('[')
    end   = s.rfind(']')
    if start != -1 and end != -1 and end > start:
        candidate = s[start:end+1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass # Fallback to other methods
    return None

# --- Core Pipeline Stages ---   

def get_transcript_smart(url, job_id, jobs):
    """Smart transcript retrieval."""
    video_id = get_video_id(url)
    if not video_id: raise ValueError("Invalid YouTube URL")

    try:
        update_job_status(job_id, jobs, 'processing', 10, 'Fetching captions...')
        transcript_list = ytt.fetch(video_id)
        segments = to_segments(transcript_list)
        return segments, video_id
    except NoTranscriptFound:
        update_job_status(job_id, jobs, 'processing', 20, 'No captions. Downloading video...')
        video_path, _ = download_audio(url, job_id)
        update_job_status(job_id, jobs, 'processing', 40, 'Transcribing with Whisper...')
        segments = transcribe_video_whisper(video_path)
        if os.path.exists(video_path): os.remove(video_path)
        return segments, video_id


#def download_video(url, job_id):
 #   """Downloads a video from YouTube."""
  #  ydl_opts = {'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4', 'outtmpl': os.path.join(UPLOAD_FOLDER, f'{job_id}.%(ext)s'), 'quiet': True}
   # with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    #    info = ydl.extract_info(url, download=True)
     #   return ydl.prepare_filename(info), info.get('title', 'video')


def _pick_audio_format(info):
    """
    Prefer English (lang startswith 'en') across ALL audio-only formats (140/251/250/249/…),
    choose the highest ABR among them. If none are English, choose highest ABR overall.
    """
    fmts = info.get("formats") or []

    # audio-only (no video)
    audio = [f for f in fmts if (f.get("vcodec") in (None, "none"))]

    def fid(f): return str(f.get("format_id", ""))
    def lang(f): return (f.get("language") or "").lower()
    def is_en(f): return lang(f).startswith("en")  # en, en-us, en-gb, etc.
    def abr_kbps(f):
        # prefer ABR; fall back to TBR if ABR missing
        a = f.get("abr")
        t = f.get("tbr")
        if isinstance(a, (int, float)) and a > 0: return a
        if isinstance(t, (int, float)) and t > 0: return t
        return -1
    def codec_pref(f):
        # tiny tie-breaker only when ABR ties: slightly prefer AAC/m4a over opus/webm
        ac = (f.get("acodec") or "").lower()
        ext = (f.get("ext") or "").lower()
        if "mp4a" in ac or ext == "m4a": return 2
        if "opus" in ac or ext == "webm": return 1
        return 0

    if not audio:
        return "bestaudio/best"

    # Rank: English first, then by ABR desc, then by codec preference
    audio_sorted = sorted(
        audio,
        key=lambda f: (1 if is_en(f) else 0, abr_kbps(f), codec_pref(f)),
        reverse=True
    )

    return fid(audio_sorted[0]) or "bestaudio/best"


def download_audio(url, job_id):

   # Download audio-only (no post-processing). Returns (path, title, video_id).   
    
    # Probe first to choose an actually-available format
    probe_opts = {
        "quiet": True,
        "noplaylist": True,
        "retries": 10,
        "fragment_retries": 10,
        "http_headers": {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"},
    }
    with yt_dlp.YoutubeDL(probe_opts) as ydl:
        info0 = ydl.extract_info(url, download=False)
        fmt = _pick_audio_format(info0)

    ydl_opts = {
        "format": fmt,  # chosen from what's actually available
        "outtmpl": os.path.join(UPLOAD_FOLDER, f"{job_id}.%(ext)s"),
        "quiet": True,
        "noplaylist": True,
        "retries": 10,
        "fragment_retries": 10,
        "concurrent_fragment_downloads": 1,  # safer vs 403s
        "http_headers": {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"},
        "overwrites": True,
        "noprogress": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)  # this will be .m4a or .webm depending on fmt
        return path, info.get("title", "video"), get_video_id(url)
    
   # Download YouTube audio without Android/PO token.
   # Copies/remuxes (no re-encode). Returns (path, title, video_id).
    
    # Probe to see what actually exists
    probe_opts = {
        "quiet": True,
        "noplaylist": True,
        "retries": 10,
        "fragment_retries": 10,
        "http_headers": {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"},
    }
    with yt_dlp.YoutubeDL(probe_opts) as ydl:
        info_probe = ydl.extract_info(url, download=False)
        fmt = _pick_audio_format(info_probe)

    ydl_opts = {
        "format": fmt,  # chosen from the real list
        "outtmpl": os.path.join(UPLOAD_FOLDER, f"{job_id}.%(ext)s"),
        "quiet": True,
        "noplaylist": True,
        "retries": 10,
        "fragment_retries": 10,
        "concurrent_fragment_downloads": 1,   # safer vs 403 bursts
        "http_headers": {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"},
        # No re-encode: copy audio; remux if needed
        "postprocessors": [{"key": "FFmpegCopyAudio"}],
        "overwrites": True,
        "noprogress": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)
        # If ffmpeg remuxes, yt-dlp will set the right extension automatically.
        return path, info.get("title", "video"), get_video_id(url)
    # Download YouTube audio with Android/PO token (often less throttled).    
    ydl_opts = {
        # prefer English m4a (140-9), then English opus (251-9), then any English, then best
        'format': '140-9/251-9/bestaudio[language=en]/bestaudio',
        'outtmpl': os.path.join(UPLOAD_FOLDER, f'{job_id}.%(ext)s'),
        'quiet': True,
        'noplaylist': True,
        'retries': 10,
        'fragment_retries': 10,
        'concurrent_fragment_downloads': 1,  # avoid 403 storms on DASH
        'http_headers': {
            'User-Agent': 'Mozilla/5.0',
            'Accept-Language': 'en-US,en;q=0.9',
        },
        # use Android client to reduce throttling / 403
        'extractor_args': {'youtube': {'player_client': ['android']}},
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)
        # normalize common input extensions to .mp3
        for ext in ('.webm', '.m4a', '.mp4', '.ogg', '.opus'):
            if path.endswith(ext):
                path = path[:-len(ext)] + '.mp3'
                break
        return path, info.get('title', 'video'), get_video_id(url)




def _ensure_wav_16k_mono(src_path: str, overwrite: bool = False) -> str:
    """
    Ensure the given audio file is a 16 kHz mono PCM WAV.
    - If src is already WAV@16k mono PCM: returns src_path.
    - Else: creates "<stem>_16k.wav" (or overwrites same name if overwrite=True) and returns it.

    Requires: ffmpeg & ffprobe in PATH.
    """
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Audio file not found: {src_path}")

    # If it's already a WAV, check properties with ffprobe
    def _is_wav_16k_mono_pcm(path: str) -> bool:
        try:
            p = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a:0",
                 "-show_entries", "stream=sample_rate,channels,codec_name",
                 "-of", "json", path],
                capture_output=True, text=True, check=True
            )
            s = (json.loads(p.stdout).get("streams") or [{}])[0]
            sr_ok  = str(s.get("sample_rate")) == "16000"
            ch_ok  = int(s.get("channels", 0)) == 1
            pcm_ok = str(s.get("codec_name", "")).startswith("pcm")
            return sr_ok and ch_ok and pcm_ok
        except Exception:
            return False  # If probe fails, we'll convert.

    if src_path.lower().endswith(".wav") and _is_wav_16k_mono_pcm(src_path):
        return src_path

    # Decide destination
    stem, _ = os.path.splitext(src_path)
    dst_path = stem + ".wav" if overwrite else stem + "_16k.wav"

    # Convert with ffmpeg: 16 kHz, mono, signed 16-bit PCM
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1",
             "-vn", "-sample_fmt", "s16", dst_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Install it and ensure it's on your PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed to convert {src_path} → {dst_path}") from e

    return dst_path

def run_diarization(audio_path):
    """Performs speaker diarization on the audio file."""
    if not HF_TOKEN:
        raise ValueError("HUGGING_FACE_TOKEN is required for pyannote.audio.")
    if not hasattr(np, "float_"):
        np.float_ = np.float64
    if not hasattr(np, "int_"):
        np.int_ = np.int64
    if not hasattr(np, "complex_"):
        np.complex_ = np.complex128
    wav_16 = _ensure_wav_16k_mono(audio_path)
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
    pipeline.to(torch.device(DEVICE))
    diarization = pipeline(wav_16)
    return diarization

def run_transcription_and_alignment(audio_path, diarization):
    """Transcribes with WhisperX and aligns with diarization results."""
    if not hasattr(np, "float_"):
        np.float_ = np.float64
    if not hasattr(np, "int_"):
        np.int_ = np.int64
    if not hasattr(np, "complex_"):
        np.complex_ = np.complex128
    audio = whisperx.load_audio(audio_path)
    model = whisperx.load_model("base.en", DEVICE, compute_type=COMPUTE_TYPE)
    result = model.transcribe(audio, batch_size=16)
    
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=DEVICE)
    aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)

    final_result = whisperx.assign_word_speakers(diarization, aligned_result)
    return final_result["segments"]

def run_transcription_only(audio_path):
    """Transcribes with WhisperX without diarization."""
    if not hasattr(np, "float_"):
        np.float_ = np.float64
    if not hasattr(np, "int_"):
        np.int_ = np.int64
    if not hasattr(np, "complex_"):
        np.complex_ = np.complex128
    audio = whisperx.load_audio(audio_path)
    model = whisperx.load_model("base.en", DEVICE, compute_type=COMPUTE_TYPE)
    result = model.transcribe(audio, batch_size=16)

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=DEVICE)
    aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)

    return aligned_result["segments"] # Return segments with word-level timestamps

def _guess_mime(path: str) -> str:
    mt = mimetypes.guess_type(path)[0]
    if mt:
        return mt
    ext = os.path.splitext(path.lower())[1]
    return {
        ".m4a": "audio/mp4",
        ".mp4": "audio/mp4",
        ".webm": "audio/webm",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
        ".opus": "audio/ogg",
    }.get(ext, "audio/mp4")

def transcribe_file_via_deepgram_to_whisperx_segments(audio_path: str):
    """
    Call Deepgram on a local audio file and return a WhisperX-like structure:
      [ { "words": [ {word,start,end,speaker?}, ... ] } ]
    so your existing sentence builders work unchanged.
    """
    if not DG_API_KEY:
        raise RuntimeError("DEEPGRAM_API_KEY not set")

    mime = _guess_mime(audio_path)
    with open(audio_path, "rb") as f:
        opts = PrerecordedOptions(
            model="nova-3",      # or "nova-3"
            diarize=False,        # include speaker IDs on words
            utterances=False,     # optional (handy for debugging)
            punctuate=True,
            paragraphs=True,    # we don't need paragraphs
            smart_format=True,
        )
        resp = DG.listen.prerecorded.v("1").transcribe_file({"buffer": f, "mimetype": mime}, opts)
        
    alt = resp.results.channels[0].alternatives[0]
    words = alt.words or []

    whisperx_like = [{
        "words": [
            {
                "word":   w["word"],
                "start":  float(w["start"]),
                "end":    float(w["end"]),
                "speaker": str(w.speaker) if getattr(w, "speaker", None) is not None else "0",
            }
            for w in words
        ]
    }]
    return whisperx_like


SENT_END_RE = re.compile(r'([\.!?]+[”"\')\]]*\s+|[\.!?]+$)')  # ., !, ?, with trailing quotes/brackets



def transcribe_file_via_deepgram(audio_path: str, model: str = "nova-3", diarize: bool = True):
    """Call Deepgram pre-recorded and return the SDK response (not a tuple)."""
    if not DG:
        raise RuntimeError("DEEPGRAM_API_KEY not set")

    mime = _guess_mime(audio_path)

    with open(audio_path, "rb") as f:
        opts = PrerecordedOptions(
            model=model,
            diarize=diarize,
            # Set to True only if you actually need utterance segments in the response:
            utterances=False,
            punctuate=True,
            paragraphs=True,
            smart_format=True,
        )
        # NOTE: removed trailing comma so we return the response, not a 1-tuple
        return DG.listen.prerecorded.v("1").transcribe_file(
            {"buffer": f, "mimetype": mime},
            opts
        )
    


def _utterance_words(utt, alt_words) -> List:
    """
    Deepgram usually returns utt.words. If not (rare), fall back by
    slicing the channel's words by utterance time window.
    """
    if getattr(utt, "words", None):
        return [w for w in utt.words if w.start is not None and w.end is not None and w.word]
    # Fallback: filter channel words by time
    return [
        w for w in alt_words
        if (w.start is not None and w.end is not None and w.word
            and w.start >= utt.start - 0.05 and w.end <= utt.end + 0.05)
    ]

def _word_char_ranges_in_transcript(transcript: str, words: List) -> List:
    """
    Build approximate char spans for each Deepgram word within the utterance transcript.
    We search sequentially (case-insensitive), tolerating punctuation/spacing differences.
    """
    ranges = []
    pos = 0
    t = transcript
    for w in words:
        token = w.word.strip()
        if not token:
            continue
        # find the next occurrence from current pos
        idx = t.lower().find(token.lower(), pos)
        if idx == -1:
            # be forgiving: try a small sliding window backtrack
            back = max(0, pos - 10)
            idx = t.lower().find(token.lower(), back)
        if idx == -1:
            # if still not found, approximate consecutive placement
            idx = pos
        start_ch = idx
        end_ch = idx + len(token)
        ranges.append((start_ch, end_ch, float(w.start), float(w.end)))
        pos = end_ch
    return ranges

def deepgram_utterances_to_sentence_segments(resp) -> List[Dict[str, Any]]:
    """
    Convert Deepgram SDK v3 pre-recorded response (with utterances=True)
    into sentence-level segments WITH punctuation and accurate timestamps.

    Returns: [{ "text": str, "start": float, "end": float, "speaker": "0/1/.." }, ...]
    """
    # Channel-wide words (for rare fallback)
    alt = resp.results.channels[0].alternatives[0]
    alt_words = [w for w in (alt.words or []) if w.start is not None and w.end is not None and w.word]

    sentence_segments: List[Dict[str, Any]] = []
    utterances = getattr(resp.results, "utterances", []) or []
    for utt in utterances:
        text = (utt.transcript or "").strip()
        if not text:
            continue

        # 1) Split this utterance's transcript into sentence-like spans by punctuation.
        spans = []
        last = 0
        for m in SENT_END_RE.finditer(text):
            spans.append((last, m.end()))
            last = m.end()
        if last < len(text):
            spans.append((last, len(text)))

        # 2) Get the words belonging to this utterance, then compute char ranges for each word.
        words = _utterance_words(utt, alt_words)
        word_char_ranges = _word_char_ranges_in_transcript(text, words) if words else []

        # 3) For each sentence span, take min(start) & max(end) over overlapping words.
        for s0, s1 in spans:
            sent_text = text[s0:s1].strip()
            if not sent_text:
                continue

            ts = [wc for wc in word_char_ranges if not (wc[1] <= s0 or wc[0] >= s1)]
            if ts:
                s_start = min(t[2] for t in ts)
                s_end   = max(t[3] for t in ts)
            else:
                # fallback to utterance time if no overlap found
                s_start, s_end = float(utt.start), float(utt.end)

            sentence_segments.append({
                "text": sent_text,
                "start": s_start,
                "end": s_end,
                "speaker": str(getattr(utt, "speaker", "0") if getattr(utt, "speaker", None) is not None else "0")
            })

    return sentence_segments

def create_speaker_aware_sentences(whisperx_segments):
    """Groups words into sentences and assigns a speaker to each sentence."""
    sentences = []
    current_sentence = {"text": "", "start": -1, "end": -1, "speaker": ""}
    
    for segment in whisperx_segments:
        if 'words' not in segment: continue
        for word_data in segment['words']:
            if 'speaker' not in word_data or 'word' not in word_data: continue

            word, start_time, end_time, speaker = word_data['word'], word_data['start'], word_data['end'], word_data['speaker']

            if current_sentence['start'] == -1:
                current_sentence['start'] = start_time
                current_sentence['speaker'] = speaker

            current_sentence['text'] += word + " "
            current_sentence['end'] = end_time

            if word.strip().endswith(('.', '?', '!')):
                current_sentence['text'] = current_sentence['text'].strip()
                sentences.append(current_sentence)
                current_sentence = {"text": "", "start": -1, "end": -1, "speaker": ""}

    if current_sentence['text']:
        current_sentence['text'] = current_sentence['text'].strip()
        sentences.append(current_sentence)
        
    return sentences
    
def calculate_diversity_scores(sequences, embedding_model):
    """Calculates a diversity score for each sequence based on its narrative reason."""
    if len(sequences) < 2:
        return [1.0] * len(sequences)

    reasons = [seq[0].get('narrative_reason', "No reason provided.") for seq in sequences]
    reason_embeddings = embedding_model.encode(reasons, convert_to_tensor=True)
    
    similarity_matrix = util.pytorch_cos_sim(reason_embeddings, reason_embeddings).cpu().numpy()
    
    diversity_scores = []
    for i in range(len(sequences)):
        avg_similarity = (np.sum(similarity_matrix[i]) - 1) / (len(sequences) - 1)
        diversity_scores.append(1 - avg_similarity)
        
    return diversity_scores

def _char_spans_for_words(text: str, words) -> List:
    spans, pos, low = [], 0, text.lower()
    for w in words:
        token = (w.word or "").strip()
        if not token: continue
        i = low.find(token.lower(), pos)
        if i == -1:
            i = low.find(token.lower(), max(0, pos-10))
        if i == -1:
            i = pos
        spans.append((i, i+len(token), float(w.start), float(w.end)))
        pos = i + len(token)
    return spans

def _split_text_to_sentences(text: str) -> List[tuple]:
    spans, last = [], 0
    for m in SENT_END_RE.finditer(text):
        spans.append((last, m.end()))
        last = m.end()
    if last < len(text):
        spans.append((last, len(text)))
    return spans

def _words_for_unit(unit, all_words):
    """Prefer unit.words; else slice channel words by unit time."""
    if getattr(unit, "words", None):
        return [w for w in unit.words if w.start is not None and w.end is not None and w.word]
    u_start = float(getattr(unit, "start", 0.0))
    u_end   = float(getattr(unit, "end",   0.0)) or (all_words[-1].end if all_words else 0.0)
    return [w for w in all_words if w.start is not None and w.end is not None and w.word
            and w.start >= u_start - 0.05 and w.end <= u_end + 0.05]

def deepgram_to_sentence_segments(resp) -> List[Dict[str, Any]]:
    """
    Prefer paragraphs (punctuated) then fallback to utterances; map sentences back to word timings.
    Returns: [{text, start, end, speaker}]
    """
    ch  = resp.results.channels[0]
    alt = ch.alternatives[0]
    all_words = [w for w in (alt.words or []) if w.start is not None and w.end is not None and w.word]

    out: List[Dict[str, Any]] = []
    paragraphs = getattr(resp.results, "paragraphs", None)
    utterances = getattr(resp.results, "utterances", None)

    units = []
    if paragraphs and getattr(paragraphs, "paragraphs", None):
        for p in paragraphs.paragraphs:
            units.append(p)
    elif utterances:
        for u in utterances:
            units.append(u)
    else:
        units.append(alt)

    for unit in units:
        text = (getattr(unit, "transcript", None) or getattr(alt, "transcript", "")).strip()
        if not text:
            continue

        words = _words_for_unit(unit, all_words)
        # if Deepgram already supplies sentence objects under paragraph, use them
        if getattr(unit, "sentences", None):
            for s in unit.sentences:
                s_text = (s.text or "").strip()
                if not s_text: continue
                w_in = [w for w in words if w.start >= float(s.start) - 0.05 and w.end <= float(s.end) + 0.05]
                if w_in:
                    s_start = float(min(w.start for w in w_in))
                    s_end   = float(max(w.end   for w in w_in))
                else:
                    s_start, s_end = float(s.start), float(s.end)
                out.append({
                    "text": s_text,
                    "start": s_start,
                    "end": s_end,
                    "speaker": str(getattr(unit, "speaker", "0") if getattr(unit, "speaker", None) is not None else "0"),
                })
            continue

        # otherwise split by punctuation and align via words
        char_map = _char_spans_for_words(text, words) if words else []
        for s0, s1 in _split_text_to_sentences(text):
            s_text = text[s0:s1].strip()
            if not s_text: continue
            overlaps = [c for c in char_map if not (c[1] <= s0 or c[0] >= s1)]
            if overlaps:
                s_start = min(c[2] for c in overlaps)
                s_end   = max(c[3] for c in overlaps)
            else:
                s_start = float(getattr(unit, "start", 0.0))
                s_end   = float(getattr(unit, "end",   0.0)) or s_start
            out.append({
                "text": s_text,
                "start": s_start,
                "end": s_end,
                "speaker": str(getattr(unit, "speaker", "0") if getattr(unit, "speaker", None) is not None else "0"),
            })
    return out

def transcribe_video_whisper(video_path):
    """Transcribes the video using Whisper."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(TRANSCRIPTION_MODEL, device=device)
    result = model.transcribe(video_path, word_timestamps=False)
    return result['segments']
    
def create_sentence_aware_segments(segments: List[Dict[str, Any]],
                                   gap_sec: float = 1.0,
                                   max_chars: int = 400) -> List[Dict[str, Any]]:
    """
    Merge YouTube transcript segments (each: {'text','start','end'}) into
    sentence-level spans with accurate start/end times from the covering segments.
    """

    def norm(t: str) -> str:
        return re.sub(r'\s+', ' ', t.strip())

    buf_text = ""                                 # rolling text buffer
    buf_map: List[Tuple[int,int,int]] = []        # (char_start, char_end, seg_idx)
    out: List[Dict[str, Any]] = []

    def flush_sentence(cut_end_off: int):
        """Emit sentence covering buf_text[0:cut_end_off] and shrink buffer."""
        nonlocal buf_text, buf_map, out
        if not buf_text or cut_end_off <= 0:
            return

        # Find first & last segment intersecting [0, cut_end_off)
        start_seg_idx = None
        end_seg_idx = None
        for s_off, e_off, seg_i in buf_map:
            if s_off < cut_end_off and e_off > 0:
                if start_seg_idx is None:
                    start_seg_idx = seg_i
                end_seg_idx = seg_i
        if start_seg_idx is None or end_seg_idx is None:
            return

        sent_text = buf_text[:cut_end_off].strip()
        if not sent_text:
            # Trim consumed portion and return
            tail = buf_text[cut_end_off:]
            ltail = tail.lstrip()
            delta = len(tail) - len(ltail)
            buf_text = ltail
            buf_map = [(s - cut_end_off - delta, e - cut_end_off - delta, i)
                       for (s, e, i) in buf_map if e > cut_end_off]
            return

        start_time = float(segments[start_seg_idx]['start'])
        end_time   = float(segments[end_seg_idx]['end'])
        out.append({"text": sent_text, "start": start_time, "end": end_time})

        # Remove consumed part from buffer (preserving correct offsets)
        tail = buf_text[cut_end_off:]
        ltail = tail.lstrip()
        delta = len(tail) - len(ltail)  # spaces trimmed by lstrip
        buf_text = ltail
        new_map = []
        for s_off, e_off, seg_i in buf_map:
            if e_off <= cut_end_off:
                continue
            new_map.append((s_off - cut_end_off - delta, e_off - cut_end_off - delta, seg_i))
        buf_map = new_map

    def append_segment(seg_text: str, seg_idx: int):
        nonlocal buf_text, buf_map
        t = norm(seg_text)
        if not t:
            return
        prefix = "" if not buf_text or buf_text.endswith(" ") else " "
        start_off = len(buf_text) + len(prefix)
        buf_text = buf_text + prefix + t
        end_off = len(buf_text)
        buf_map.append((start_off, end_off, seg_idx))

    prev_end = None
    for i, seg in enumerate(segments):
        start = float(seg["start"])
        end   = float(seg.get("end", start))
        if prev_end is not None and (start - prev_end) >= gap_sec and buf_text:
            # large silence ⇒ flush whatever we have
            flush_sentence(len(buf_text))

        append_segment(seg["text"], i)
        prev_end = end

        # punctuation-based flush: up to the last sentence end we can find
        last_cut = None
        for m in SENT_END_RE.finditer(buf_text):
            last_cut = m.end()
        if last_cut:
            flush_sentence(last_cut)

        # safety: avoid gigantic “sentences” when no punctuation appears
        if len(buf_text) >= max_chars:
            flush_sentence(len(buf_text))

    # tail
    if buf_text:
        flush_sentence(len(buf_text))

    return out

def create_sentence_segments_from_whisperx(whisperx_segments):
    """Groups words into sentences from a non-diarized WhisperX transcript."""
    sentences = []
    current_sentence = {"text": "", "start": -1, "end": -1}
    
    for segment in whisperx_segments:
        if 'words' not in segment: continue
        for word_data in segment['words']:
            if 'word' not in word_data: continue

            word, start_time, end_time = word_data['word'], word_data['start'], word_data['end']

            if current_sentence['start'] == -1:
                current_sentence['start'] = start_time

            current_sentence['text'] += word + " "
            current_sentence['end'] = end_time

            if word.strip().endswith(('.', '?', '!')):
                current_sentence['text'] = current_sentence['text'].strip()
                sentences.append(current_sentence)
                current_sentence = {"text": "", "start": -1, "end": -1}

    if current_sentence['text']:
        current_sentence['text'] = current_sentence['text'].strip()
        sentences.append(current_sentence)
        
    return sentences

def identify_sequences(transcript_sentences, relevant_indices_set): 
    """Groups consecutive relevant sentences into sequences."""
    sequences = []
    current_sequence = []
    for i, sentence in enumerate(transcript_sentences):
        if i in relevant_indices_set:
            current_sequence.append(sentence)
        else:
            if current_sequence:
                sequences.append(current_sequence)
                current_sequence = []
    if current_sequence:
        sequences.append(current_sequence)
    return sequences

def score_sequences(sequences, score_by_id, diversity_scores):
    """Assigns a value score to each sequence based on relevance and diversity."""
    scored_sequences = []
    for i, seq in enumerate(sequences):
        seq_indices = [s['original_index'] for s in seq]
        seq_sims = [score_by_id.get(j, 0.0) for j in seq_indices]   # ← safe
        if seq_sims:
            avg_relevance  = float(np.mean(seq_sims))
            peak_relevance = float(np.max(seq_sims))
        else:
            avg_relevance = peak_relevance = 0.0
        duration = seq[-1]['end'] - seq[0]['start']
        
        relevance_score = (avg_relevance * 0.7 + peak_relevance * 0.3)
        combined_score = relevance_score * 0.6 + diversity_scores[i] * 0.4
        value = combined_score * (1 + 0.01 * duration)
        
        scored_sequences.append({
            "sentences": seq, "duration": int(duration), "value": int(value * 100),
            "start": seq[0]['start'], "end": seq[-1]['end'], "narrative_reason": seq[0].get('narrative_reason', '')
        })
    return scored_sequences

def infer_duration(query):
    """Infers an optimal duration based on the query type (Specific vs. Broad)."""
    messages = [
        {"role": "user", "content": f"""Analyze the user's query to determine if it is 'Specific' or 'Broad'.
- A 'Specific' query asks for a single fact, moment, or definition.
- A 'Broad' query asks for a general overview, summary, or collection of ideas.
Based on this, suggest an ideal summary length in seconds.
- For 'Specific' queries, suggest a max duration upto 45 seconds.
- For 'Broad' queries, suggest a max duration upto 390 seconds.
Respond with a JSON object containing "type" and "suggested_duration_seconds".

Query: "{query}"
"""}
    ]
    response = generate_gemma_response(messages, NARRATIVE_MODEL)
    try:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return data.get("suggested_duration_seconds", 180)
        return 180
    except (json.JSONDecodeError, TypeError):
        return 180

def knapsack_selection(sequences, capacity):
    """Selects the optimal combination of sequences to maximize value."""
    n = len(sequences)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            weight = sequences[i-1]['duration']
            value = sequences[i-1]['value']
            if weight <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-weight] + value)
            else:
                dp[i][w] = dp[i-1][w]
    
    selected_sequences = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected_sequences.append(sequences[i-1])
            w -= sequences[i-1]['duration']
            
    return selected_sequences[::-1]

def generate_narrative_reasons(query, selected_sequences):
    """Uses an LLM to provide a one-sentence reason for each selected sequence."""
    clips_for_prompt = [{"text": " ".join(s['text'] for s in seq['sentences']), "start": seq['start'], "end": seq['end']} for seq in selected_sequences]

    messages = [
        {"role": "user", "content": f"""You are a research assistant. Your task is to provide a brief, one-sentence reason why each of the following video clips is relevant to the user's original query: "{query}"
You will be given a list of clips. For each clip, provide a concise "narrative_reason".
RULES:
1. The reason should be a single, clear sentence.
2. The output MUST be a valid JSON array of objects, preserving the original "start" and "end" times and adding your "narrative_reason".

Here are the selected clips:
{json.dumps(clips_for_prompt, indent=2)}

Generate the final JSON script:
"""}
    ]
    response_text = generate_gemma_response(messages, NARRATIVE_MODEL)
    
    playlist = extract_json_array(response_text)
    if playlist:
        return playlist
    
    # Fallback if JSON is invalid
    return [{"start": seq['start'], "end": seq['end'], "narrative_reason": "This segment provides relevant context."} for seq in selected_sequences]




def clean_query_for_search(query): 
    """
    Uses an LLM to extract a duration and return a 'clean' query
    focused only on the core topic for better semantic search.
    """
    messages = [
        {"role": "user", "content": f"""Analyze the following user query. Your task is to separate the core topic from any duration-related commands.
         Respond with a JSON object containing two keys: "duration_minutes" and "clean_query". If there is no duration mentioned, "duration_minutes" should be null.
         - "duration_minutes": An integer representing the requested time in minutes. If no duration is mentioned, this should be null.
         - "clean_query": The user's query with all duration-related phrases removed, focusing only on the main topic.

Examples:
- Query: "give me a 5 minute summary about the economy" -> Response: {{"duration_minutes": 5, "clean_query": "a summary about the economy"}}
- Query: "show me the highlights" -> Response: {{"duration_minutes": null, "clean_query": "show me the highlights"}}
- Query: "a 2-minute clip of what he said about AI" -> Response: {{"duration_minutes": 2, "clean_query": "what he said about AI"}}

Query: "{query}"
"""}
    ]
    response = generate_gemma_response(messages, NARRATIVE_MODEL)
    try:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return data.get("duration_minutes"), data.get("clean_query", query)
        return None, query
    except (json.JSONDecodeError, TypeError):
        return None, query
    

def generate_narrative_reasons_batch(query, sequences):
    """Generates narrative reasons for a batch of sequences."""
    for seq in sequences:
        full_text = " ".join([s['text'] for s in seq])
        messages = [{"role": "user", "content": f"""In one sentence, explain why the following text is relevant to the user's query. Query: "{query}" Text: "{full_text}" Reason: """}]
        reason = generate_gemma_response(messages, NARRATIVE_MODEL)
        seq[0]['narrative_reason'] = reason
    return sequences


# ---- Redis connection ----
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
# decode_responses=False so we can store/read raw bytes safely
r = redis.Redis.from_url(REDIS_URL, decode_responses=False)
r.ping()
print("Successfully connected to Redis! ✅")

VIDEO_CACHE_TTL = int(os.getenv("VIDEO_CACHE_TTL", "604800"))  # 7 days



# ------- HSET manual mapping=... to avoid arg count errors ------- #
# def _hset_mapping(conn: redis.Redis, name: str, mapping: dict):
#     """
#     Robust HSET that works across redis-py versions and avoids 'wrong number of arguments'.
#     Falls back to manual flattening if needed.
#     """
#     if not mapping:
#         return 0

#     # 1) Try modern API (redis-py >= 4): hset(name, mapping=...)
#     try:
#         return conn.hset(name, mapping=mapping)
#     except TypeError:
#         # Old client that doesn't accept mapping=; try HMSET if available
#         try:
#             return conn.hmset(name, mapping)  # deprecated on server, but works as compatibility
#         except AttributeError:
#             pass  # fall through to manual flatten

#     # 2) Manual flatten (works with any redis-py version and pipelines)
#     flat = []
#     for field, value in mapping.items():
#         # Field must be bytes or str
#         if isinstance(field, str):
#             field = field.encode("utf-8")
#         elif not isinstance(field, (bytes, bytearray)):
#             field = bytes(field)

#         # Value must be bytes (your serializers already ensure bytes)
#         if isinstance(value, memoryview):
#             value = bytes(value)
#         elif isinstance(value, str):
#             value = value.encode("utf-8")
#         elif not isinstance(value, (bytes, bytearray)):
#             # Safety: last-resort cast
#             value = bytes(value)

#         flat.extend([field, value])

#     return conn.execute_command("HSET", name, *flat)

def _hset_fields_safely(conn, name: str, mapping: dict, ttl: int = None):
    """
    Set hash fields one-by-one in a pipeline.
    Eliminates all HSET arity/version issues.
    """
    if not mapping:
        return

    def _to_bytes(x):
        if isinstance(x, (bytes, bytearray)): return bytes(x)
        if isinstance(x, memoryview): return bytes(x)
        if isinstance(x, str): return x.encode("utf-8")
        raise TypeError(f"value must be bytes/str, got {type(x).__name__}")

    with conn.pipeline() as pipe:
        for field, value in mapping.items():
            # field: bytes
            if isinstance(field, (bytes, bytearray)):
                f = bytes(field)
            else:
                f = str(field).encode("utf-8")
            v = _to_bytes(value)
            pipe.hset(name, f, v)         # HSET key field value (always valid)
        if ttl is not None:
            pipe.expire(name, ttl)
        pipe.execute()

# -------- Debugging helpers -------- #
def _debug_check_mapping(mapping: dict):
    for k, v in mapping.items():
        if v is None:
            raise ValueError(f"Redis mapping value for {k!r} is None")
        if not isinstance(v, (bytes, bytearray, memoryview, str)):
            # Our serializers should have produced bytes already
            raise TypeError(f"Redis mapping value for {k!r} must be bytes/str, got {type(v).__name__}")

# ---- Serialization helpers ----
def _dumps_json(obj) -> bytes:
    return orjson.dumps(obj)

def _loads_json(buf: bytes):
    return orjson.loads(buf)

def _dumps_ndarray(arr: np.ndarray) -> bytes:
    # ensure contiguous & desired dtype to avoid surprises
    arr = np.ascontiguousarray(arr).astype(np.float32, copy=False)
    bio = io.BytesIO()
    # NPY header + data (portable); then compress to shrink Redis memory
    np.save(bio, arr, allow_pickle=False)
    out = zlib.compress(bio.getvalue(), level=6)
    assert isinstance(out, (bytes, bytearray)), "ndarray dump must be bytes"
    return out

def _loads_ndarray(buf: bytes) -> np.ndarray:
    raw = zlib.decompress(buf)
    return np.load(io.BytesIO(raw), allow_pickle=False)

def _dumps_faiss_index(index) -> bytes:
    """
    Robustly turn a FAISS index into bytes across FAISS versions.
    """
    # 1) Try in-memory serialization
    try:
        vec = faiss.serialize_index(index)  # VectorUint8 or bytes depending on build
        # Already bytes?
        if isinstance(vec, (bytes, bytearray, memoryview)):
            return bytes(vec)
        # VectorUint8 -> numpy array -> bytes
        if hasattr(faiss, "vector_to_array"):
            arr = faiss.vector_to_array(vec)           # np.ndarray dtype=uint8
            return arr.tobytes()
        # Some builds may let numpy view work directly
        try:
            arr = np.asarray(vec, dtype=np.uint8)
            return arr.tobytes()
        except Exception:
            pass
    except Exception:
        pass

    # 2) Fallback: write to a temp file, then read the bytes
    tmp_path = os.path.join(JOB_DATA_FOLDER, f"_tmp_{uuid.uuid4().hex}.index")
    faiss.write_index(index, tmp_path)
    with open(tmp_path, "rb") as f:
        data = f.read()
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    return data

def _loads_faiss_index(buf: bytes):
    """
    Rebuild a FAISS index from bytes, using the fastest path available.
    """
    # 1) Try direct in-memory deserialization
    try:
        # Many builds accept a numpy uint8 array directly
        arr = np.frombuffer(buf, dtype=np.uint8)
        return faiss.deserialize_index(arr)
    except Exception:
        pass

    # 2) Fallback: write to a temp file, then read it
    tmp_path = os.path.join(JOB_DATA_FOLDER, f"_tmp_{uuid.uuid4().hex}.index")
    with open(tmp_path, "wb") as f:
        f.write(buf)
    try:
        idx = faiss.read_index(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return idx

def _cache_key(video_id: str) -> str:
    return f"{video_id}"

def _fetch_cached_video(video_id: str):
    """Return dict with objects if ALL fields are present and valid; else None."""
    key = _cache_key(video_id)

    def _hget_dual(name: str):
        # Try str then bytes, then bytes then str (covers how fields were stored)
        v = r.hget(key, name)
        if v is None:
            v = r.hget(key, name.encode("utf-8"))
        if v is None:
            # In case caller passed bytes already
            try:
                v = r.hget(key, name.decode("utf-8"))  # will fail if name is str
            except Exception:
                pass
        return v

    # Fetch fields one by one
    ss  = _hget_dual("sentence_segments")
    tx  = _hget_dual("texts")
    emb = _hget_dual("embeddings")
    fx  = _hget_dual("faiss_index")
    mt  = _hget_dual("meta")

    # If any is missing, treat as miss
    if any(x is None for x in (ss, tx, emb, fx, mt)):
        # Optional debug: list existing fields so you can see what's there
        try:
            existing = r.hkeys(key)
            try:
                existing = [e.decode() if isinstance(e, (bytes, bytearray)) else str(e) for e in existing]
            except Exception:
                existing = [str(e) for e in existing]
            print(f"[CACHE MISS] video_id={video_id} key={key} — found fields: {existing}")
        except Exception:
            pass
        return None

    # Deserialize
    try:
        sentence_segments = _loads_json(ss)
        texts             = _loads_json(tx)
        embeddings        = _loads_ndarray(emb)
        index             = _loads_faiss_index(fx)
        meta              = _loads_json(mt)
    except Exception as e:
        print(f"[CACHE CORRUPT] video_id={video_id} key={key} — {e}")
        return None

    # Validate
    if meta.get("embedding_model") != EMBEDDING_MODEL:
        print(f"[CACHE INVALID] video_id={video_id} — model mismatch: {meta.get('embedding_model')} != {EMBEDDING_MODEL}")
        return None
    if getattr(index, "d", None) != int(embeddings.shape[1]):
        print(f"[CACHE INVALID] video_id={video_id} — dim mismatch: index.d={getattr(index,'d',None)} emb_dim={int(embeddings.shape[1])}")
        return None

    # Refresh TTL on hit
    try:
        r.expire(key, VIDEO_CACHE_TTL)
    except Exception:
        pass

    print(f"[CACHE HIT] video_id={video_id} n_sentences={meta.get('n_sentences','?')} emb_dim={meta.get('emb_dim','?')}")
    return {
        "sentence_segments": sentence_segments,
        "texts": texts,
        "embeddings": embeddings,
        "index": index,
        "meta": meta,
    }


def _store_cached_video(video_id: str, sentence_segments, texts, embeddings, index):
    key = _cache_key(video_id)
    meta = {
        "embedding_model": EMBEDDING_MODEL,
        "emb_dim": int(embeddings.shape[1]),
        "n_sentences": int(embeddings.shape[0]),
        "created_at": int(time.time()),
    }
    mapping = {
        b"sentence_segments": _dumps_json(sentence_segments),
        b"texts": _dumps_json(texts),
        b"embeddings": _dumps_ndarray(embeddings),
        b"faiss_index": _dumps_faiss_index(index),
        b"meta": _dumps_json(meta)
    }

    # print("HSET fields:", {k.decode() if isinstance(k, (bytes, bytearray)) else k: type(v).__name__ 
    #                    for k, v in mapping.items()})
    _debug_check_mapping(mapping)
    # IMPORTANT: use mapping= to avoid "wrong number of arguments for 'hset'"
    _hset_fields_safely(r, key, mapping, ttl=VIDEO_CACHE_TTL)
    r.expire(key, VIDEO_CACHE_TTL)



def process_audio_pipeline(youtube_url, query, job_id, jobs, with_narration):
    """The new, audio-first main pipeline with Redis caching."""
    try:
        update_job_status(job_id, jobs, 'processing', 10, 'Extracting audio...')
        audio_path, _, video_id = download_audio(youtube_url, job_id)

        cache_hit = _fetch_cached_video(video_id)

        if cache_hit:
            print(f"[CACHE HIT] video_id={video_id} — skipping transcription & embedding")
            update_job_status(job_id, jobs, 'processing', 65, 'Cache hit — loading embeddings & index...')
            # ---- Cache HIT: reconstruct everything needed to continue at ~65% ----
            update_job_status(job_id, jobs, 'processing', 65, 'Cache hit — loading embeddings & index...')
            sentence_segments = cache_hit["sentence_segments"]
            texts             = cache_hit["texts"]
            embeddings        = cache_hit["embeddings"]
            index             = cache_hit["index"]

            # Recreate artifacts used later in the pipeline (for downstream functions):
            for i, seg in enumerate(sentence_segments):
                seg['original_index'] = seg.get('original_index', i)

            # Write full transcript and FAISS index to disk to satisfy later consumers
            full_transcript_path = os.path.join(JOB_DATA_FOLDER, f'{job_id}_full_transcript.txt')
            with open(full_transcript_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(texts))

            index_path = os.path.join(JOB_DATA_FOLDER, f'{job_id}.index')
            faiss.write_index(index, index_path)

        else:
            print(f"[CACHE MISS] video_id={video_id} — running transcription & embedding")
            # ---- Cache MISS: run your normal pipeline until the index is ready ----
            update_job_status(job_id, jobs, 'processing', 40, 'Transcribing (Deepgram)...')
            dg_resp = transcribe_file_via_deepgram(audio_path, model="nova-3", diarize=True)

            update_job_status(job_id, jobs, "processing", 65, "Building sentence segments...")
            sentence_segments = deepgram_to_sentence_segments(dg_resp)
            texts = [s['text'] for s in sentence_segments]
            for i, seg in enumerate(sentence_segments):
                seg['original_index'] = i

            full_transcript_path = os.path.join(JOB_DATA_FOLDER, f'{job_id}_full_transcript.txt')
            with open(full_transcript_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(texts))

            # Build embeddings + FAISS
            update_job_status(job_id, jobs, 'processing', 65, 'Embedding transcript...')
            model = SentenceTransformer(EMBEDDING_MODEL)

            embeddings = model.encode(
                texts,
                batch_size=64,
                normalize_embeddings=True,  # IMPORTANT: IP == cosine later
                show_progress_bar=False
            ).astype('float32')

            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)

            index_path = os.path.join(JOB_DATA_FOLDER, f'{job_id}.index')
            faiss.write_index(index, index_path)

            # Store in Redis for next time
            _store_cached_video(video_id, sentence_segments, texts, embeddings, index)

        # ---- Query-dependent section (runs for both cache-hit & cache-miss) ----
        update_job_status(job_id, jobs, 'processing', 70, 'Searching for relevant content...')
        desired_duration_minutes, clean_query = clean_query_for_search(query)

        # We still need the model to embed the query; cheap vs. encoding all sentences
        model = SentenceTransformer(EMBEDDING_MODEL)
        query_vec = model.encode([clean_query], normalize_embeddings=True).astype('float32')

        k_first_stage = min(max(100, int(len(texts)*0.2)), len(texts))
        distances, indices = index.search(query_vec, k_first_stage)
        similarity_scores = distances[0]
        cand_ids = indices[0]
        mu, sigma = float(similarity_scores.mean()), float(similarity_scores.std())
        cut = max(mu - 0.25*sigma, 0.15)  # gentle floor
        relevant_ids = [i for i, s in zip(cand_ids, similarity_scores) if s >= cut]
        relevant_indices_set = set(relevant_ids)

        update_job_status(job_id, jobs, 'processing', 75, 'Identifying key sequences...')
        sequences = identify_sequences(sentence_segments, relevant_indices_set)

        update_job_status(job_id, jobs, 'processing', 80, 'Generating reasons for diversity ranking...')
        sequences_with_reasons = generate_narrative_reasons_batch(clean_query, sequences)
        diversity_scores = calculate_diversity_scores(sequences_with_reasons, model)
        score_by_id = {doc_id: sim for doc_id, sim in zip(cand_ids, similarity_scores)}
        scored_sequences = score_sequences(sequences_with_reasons, score_by_id, diversity_scores)

        if desired_duration_minutes == 0:
            desired_duration_minutes = None
        update_job_status(job_id, jobs, 'processing', 85, 'Inferring optimal duration...')
        target_duration = infer_duration(query) if not desired_duration_minutes else desired_duration_minutes * 60

        update_job_status(job_id, jobs, 'processing', 90, 'Selecting best sequences...')
        selected_sequences = knapsack_selection(scored_sequences, target_duration)
        if not selected_sequences:
            selected_sequences = sorted(scored_sequences, key=lambda x: x['value'], reverse=True)[:5]

        playlist = [{"start": seq['start'], "end": seq['end'], "narrative_reason": seq['narrative_reason']} for seq in selected_sequences]
        if not playlist:
            raise ValueError("Narrative Engine failed to produce a playlist.")

        narration_url = None  # Narration disabled for now

        result_data = {"video_id": video_id, "playlist": playlist, "narration_url": narration_url}
        jobs[job_id] = {'status': 'completed', 'progress': 100, 'message': 'Processing complete!', 'result': result_data}

    except Exception as e:
        print(f"Error in job {job_id}: {e}")
        jobs[job_id] = {'status': 'failed', 'progress': 100, 'message': str(e)}
    finally:
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.remove(audio_path)


# --- Main Pipeline & Q&A Functions ---

# def process_audio_pipeline(youtube_url, query, job_id, jobs, with_narration):
#     """The new, audio-first main pipeline."""
#     try:
#         update_job_status(job_id, jobs, 'processing', 10, 'Extracting audio...')
#         audio_path, _, video_id = download_audio(youtube_url, job_id)

#        ## update_job_status(job_id, jobs, 'processing', 25, 'Identifying speakers...')
#        ## diarization = run_diarization(audio_path)

# ##        update_job_status(job_id, jobs, 'processing', 50, 'Transcribing and aligning...')
# ##        whisperx_segments = run_transcription_and_alignment(audio_path, diarization)

#         ## update_job_status(job_id, jobs, 'processing', 60, 'Structuring transcript...')
#         ## sentence_segments = create_speaker_aware_sentences(whisperx_segments)
        
#         update_job_status(job_id, jobs, 'processing', 40, 'Transcribing (Deepgram)...')
#         dg_resp = transcribe_file_via_deepgram(audio_path, model="nova-3", diarize=True)


#         update_job_status(job_id, jobs, "processing", 65, "Building sentence segments...")
#         sentence_segments = deepgram_to_sentence_segments(dg_resp)
    
#         for i, seg in enumerate(sentence_segments): seg['original_index'] = i
#         full_transcript_path = os.path.join(JOB_DATA_FOLDER, f'{job_id}_full_transcript.txt')
#         with open(full_transcript_path, 'w', encoding='utf-8') as f: f.write("\n".join([s['text'] for s in sentence_segments]))
        
#        # 65% — build sentence embeddings & FAISS index first (query-independent)
#         update_job_status(job_id, jobs, 'processing', 65, 'Embedding transcript...')
#         model = SentenceTransformer(EMBEDDING_MODEL)

#         texts = [s['text'] for s in sentence_segments]
#         embeddings = model.encode(
#             texts,
#             batch_size=64,
#             normalize_embeddings=True,  # IMPORTANT: IP == cosine later
#             show_progress_bar=False
#         ).astype('float32')

#         index = faiss.IndexFlatIP(embeddings.shape[1])
#         index.add(embeddings)

#         index_path = os.path.join(JOB_DATA_FOLDER, f'{job_id}.index')
#         faiss.write_index(index, index_path)

#         # 70% — now do the query-dependent bits (cleaning + query embedding + search)
#         update_job_status(job_id, jobs, 'processing', 70, 'Searching for relevant content...')
#         desired_duration_minutes, clean_query = clean_query_for_search(query)

#         query_vec = model.encode([clean_query], normalize_embeddings=True).astype('float32')

#         k_first_stage = min( max(100, int(len(texts)*0.2)), len(texts) )
#         distances, indices = index.search(query_vec, k_first_stage)
#         similarity_scores = distances[0]
#         cand_ids = indices[0]
#         mu, sigma = float(similarity_scores.mean()), float(similarity_scores.std())
#         cut = max(mu - 0.25*sigma, 0.15)  # gentle floor
#         relevant_ids = [i for i, s in zip(cand_ids, similarity_scores) if s >= cut]
#         relevant_indices_set = set(relevant_ids)

#         update_job_status(job_id, jobs, 'processing', 75, 'Identifying key sequences...')
#         sequences = identify_sequences(sentence_segments, relevant_indices_set)

#         update_job_status(job_id, jobs, 'processing', 80, 'Generating reasons for diversity ranking...')
#         sequences_with_reasons = generate_narrative_reasons_batch(clean_query, sequences)
#         diversity_scores = calculate_diversity_scores(sequences_with_reasons, model)
#         score_by_id = {doc_id: sim for doc_id, sim in zip(cand_ids, similarity_scores)}
#         scored_sequences = score_sequences(sequences_with_reasons, score_by_id, diversity_scores)

#         if desired_duration_minutes == 0:
#             desired_duration_minutes = None
#         update_job_status(job_id, jobs, 'processing', 85, 'Inferring optimal duration...')
#         target_duration = infer_duration(query) if not desired_duration_minutes else desired_duration_minutes * 60

#         update_job_status(job_id, jobs, 'processing', 90, 'Selecting best sequences...')
#         selected_sequences = knapsack_selection(scored_sequences, target_duration)
#         if not selected_sequences: selected_sequences = sorted(scored_sequences, key=lambda x: x['value'], reverse=True)[:5]

#         # Final playlist construction
#         playlist = [{"start": seq['start'], "end": seq['end'], "narrative_reason": seq['narrative_reason']} for seq in selected_sequences]
#         if not playlist: raise ValueError("Narrative Engine failed to produce a playlist.")

#         narration_url = None # Narration disabled for now
        
#         result_data = {"video_id": video_id, "playlist": playlist, "narration_url": narration_url}
#         jobs[job_id] = {'status': 'completed', 'progress': 100, 'message': 'Processing complete!', 'result': result_data}

#     except Exception as e:
#         print(f"Error in job {job_id}: {e}")
#         jobs[job_id] = {'status': 'failed', 'progress': 100, 'message': str(e)}
#     finally:
#         if 'audio_path' in locals() and os.path.exists(audio_path):
#             os.remove(audio_path)

def answer_question_from_video(question, job_id):
    """Answers a user's question using the pre-processed video data."""
    index_path = os.path.join(JOB_DATA_FOLDER, f'{job_id}.index')
    full_transcript_path = os.path.join(JOB_DATA_FOLDER, f'{job_id}_full_transcript.txt')

    if not os.path.exists(index_path) or not os.path.exists(full_transcript_path):
        raise FileNotFoundError("Processed video data not found.")

    with open(full_transcript_path, 'r', encoding='utf-8') as f:
        full_transcript = f.read()

    index = faiss.read_index(index_path)
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode([question]).astype('float32')
    _, indices = index.search(query_embedding, k=5)
    
    sentences = full_transcript.splitlines()
    relevant_chunks = "\n".join([sentences[i] for i in indices[0] if i < len(sentences)])

    messages = [
        {"role": "user", "content": f"""You are a Q&A assistant. Answer the user's question based on the provided video transcript.
First, use the "Relevant Excerpts" for the most direct answer.
Then, use the "Full Transcript" for broader context if needed.

**Relevant Excerpts:**
{relevant_chunks}

**Full Transcript:**
{full_transcript[:12000]} 

**Question:**
{question}

**Answer:**
"""}
    ]
    return generate_gemma_response(messages, QA_MODEL)
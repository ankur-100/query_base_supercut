import os
import yt_dlp
import whisper
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import re, json, ast
from typing import List, Dict, Any, Tuple
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
import nltk
from gtts import gTTS
from dotenv import load_dotenv

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
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
NARRATIVE_MODEL = "google/gemma-3-270m-it"
QA_MODEL = "google/gemma-3-270m-it"
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
NUM_RE  = r'-?\d+(?:\.\d+)?'
SENT_END_RE = re.compile(r'([.!?…])(\s+|$)')  # sentence boundary

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
        video_path, _ = download_video(url, job_id)
        update_job_status(job_id, jobs, 'processing', 40, 'Transcribing with Whisper...')
        segments = transcribe_video_whisper(video_path)
        if os.path.exists(video_path): os.remove(video_path)
        return segments, video_id

def download_video(url, job_id):
    """Downloads a video from YouTube."""
    ydl_opts = {'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4', 'outtmpl': os.path.join(UPLOAD_FOLDER, f'{job_id}.%(ext)s'), 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info), info.get('title', 'video')

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

def score_sequences(sequences, similarity_scores):
    """Assigns a value score to each sequence."""
    scored_sequences = []
    for seq in sequences:
        seq_indices = [s['original_index'] for s in seq]
        avg_relevance = np.mean([similarity_scores[i] for i in seq_indices])
        peak_relevance = np.max([similarity_scores[i] for i in seq_indices])
        duration = seq[-1]['end'] - seq[0]['start']
        
        score = (avg_relevance * 0.7 + peak_relevance * 0.3) * (1 + 0.01 * duration)
        
        scored_sequences.append({
            "sentences": seq,
            "duration": int(duration),
            "value": int(score * 100),
            "start": seq[0]['start'],
            "end": seq[-1]['end']
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
         Respond with a JSON object containing two keys: "duration_minutes" and "clean_query".
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
    
# --- Main Pipeline & Q&A Functions ---

def process_transcript_pipeline(youtube_url, query, job_id, jobs, with_narration):
    """The main end-to-end function that orchestrates all stages."""
    try:
        segments, video_id = get_transcript_smart(youtube_url, job_id, jobs)
        
        update_job_status(job_id, jobs, 'processing', 45, 'Structuring transcript...')
        sentence_segments = create_sentence_aware_segments(segments)
        for i, seg in enumerate(sentence_segments): seg['original_index'] = i

        full_transcript_path = os.path.join(JOB_DATA_FOLDER, f'{job_id}_full_transcript.txt')
        with open(full_transcript_path, 'w', encoding='utf-8') as f:
            f.write("\n".join([s['text'] for s in sentence_segments]))
        
        update_job_status(job_id, jobs, 'processing', 60, 'Cleaning query for search...')
        _, clean_query = clean_query_for_search(query)

        update_job_status(job_id, jobs, 'processing', 65, 'Searching for relevant content...')
        model = SentenceTransformer(EMBEDDING_MODEL)
        embeddings = model.encode([s['text'] for s in sentence_segments]).astype('float32')
        query_embedding = model.encode([clean_query]).astype('float32')
        
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        index_path = os.path.join(JOB_DATA_FOLDER, f'{job_id}.index')
        faiss.write_index(index, index_path)

        distances, indices = index.search(query_embedding, k=len(sentence_segments))
        similarity_scores = 1 / (1 + distances[0])
        
        relevance_threshold = 0.5 
        relevant_indices_set = {i for i, score in zip(indices[0], similarity_scores) if score > relevance_threshold}

        update_job_status(job_id, jobs, 'processing', 70, 'Identifying key sequences...')
        sequences = identify_sequences(sentence_segments, relevant_indices_set)

        scored_sequences = score_sequences(sequences, similarity_scores)

        update_job_status(job_id, jobs, 'processing', 80, 'Inferring optimal duration...')
        target_duration = infer_duration(query)

        update_job_status(job_id, jobs, 'processing', 85, 'Selecting best sequences...')
        selected_sequences = knapsack_selection(scored_sequences, target_duration)
        if not selected_sequences:
            selected_sequences = sorted(scored_sequences, key=lambda x: x['value'], reverse=True)[:5]

        update_job_status(job_id, jobs, 'processing', 90, 'Generating final narrative...')
        playlist = generate_narrative_reasons(query, selected_sequences)
        if not playlist: raise ValueError("Narrative Engine failed to produce a playlist.")

        narration_url = youtube_url
        # Narration is disabled in this version for simplicity, can be re-added
        # if with_narration:
        #     update_job_status(job_id, jobs, 'processing', 95, 'Generating AI narration...')
        #     narration_url = generate_narration_audio(query, playlist, job_id)

        result_data = {"video_id": video_id, "playlist": playlist, "narration_url": narration_url}
        jobs[job_id] = {'status': 'completed', 'progress': 100, 'message': 'Processing complete!', 'result': result_data}

    except Exception as e:
        print(f"Error in job {job_id}: {e}")
        jobs[job_id] = {'status': 'failed', 'progress': 100, 'message': str(e)}

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

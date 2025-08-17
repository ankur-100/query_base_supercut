import os
import yt_dlp
import whisper
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips, TextClip
from moviepy.audio.fx.all import audio_fadein, audio_fadeout, volumex
from gtts import gTTS
import re
import json
import uuid
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
import os, shutil
from moviepy.config import change_settings

# Try PATH first
magick = shutil.which("magick")
if magick:
    change_settings({"IMAGEMAGICK_BINARY": magick})
else:
    # Fallback: hardcode the usual install path (adjust if needed)
    hardcoded = r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"
    if os.path.exists(hardcoded):
        change_settings({"IMAGEMAGICK_BINARY": hardcoded})
    else:
        raise RuntimeError(
            "ImageMagick not found. Install it or set IMAGEMAGICK_BINARY to magick.exe"
        )

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
JOB_DATA_FOLDER = 'job_data' # For storing indexes and transcripts
TRANSCRIPTION_MODEL = "base.en"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
NARRATIVE_MODEL = "google/flan-t5-large"
QA_MODEL = "google/flan-t5-large"

ytt = YouTubeTranscriptApi()


# --- Helper Functions ---

def update_job_status(job_id, jobs, status, progress, message):
    """Updates the shared job status dictionary."""
    if job_id in jobs:
        current_job = jobs[job_id]
        current_job['status'] = status
        current_job['progress'] = progress
        current_job['message'] = message
        jobs[job_id] = current_job

def sanitize_filename(name):
    """Removes invalid characters from a filename."""
    return re.sub(r'[\\/*?:"<>|]', "", name)

def get_video_id(url):
    """Extracts the YouTube video ID from a URL."""
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return None

# --- Core Pipeline Stages ---

def download_video(url, job_id):
    """Downloads a video from YouTube."""
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': os.path.join(UPLOAD_FOLDER, f'{job_id}.%(ext)s'),
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        return filename, info.get('title', 'video')

def transcribe_video_whisper(video_path, job_id):
    """Transcribes the video using Whisper and saves the segments."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(TRANSCRIPTION_MODEL, device=device)
    result = model.transcribe(video_path, word_timestamps=False)
    
    segments_path = os.path.join(JOB_DATA_FOLDER, f'{job_id}_segments.json')
    with open(segments_path, 'w') as f:
        json.dump(result['segments'], f)
        
    return result['segments']

def create_semantic_index(segments, job_id):
    """Creates a FAISS index from transcript segments and saves it."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    sentences = [segment['text'] for segment in segments]
    
    valid_sentences = [s for s in sentences if s.strip()]
    valid_indices = [i for i, s in enumerate(sentences) if s.strip()]
    
    if not valid_sentences:
        return None, None, [], []

    embeddings = model.encode(valid_sentences, convert_to_tensor=True).cpu().numpy()
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    index_path = os.path.join(JOB_DATA_FOLDER, f'{job_id}.index')
    faiss.write_index(index, index_path)

    return index, model, valid_sentences, valid_indices

def search_index(query, index, model, segments, valid_indices, top_k=8):
    """Searches the index for the most relevant segments."""
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
    _, indices = index.search(query_embedding, top_k)
    
    relevant_segments = []
    for i in indices[0]:
        if i < len(valid_indices):
            original_segment_index = valid_indices[i]
            segment = segments[original_segment_index]
            relevant_segments.append({
                "text": segment['text'],
                "start": segment['start'],
                "end": segment['end']
            })
    return relevant_segments

def generate_narrative_script(query, relevant_segments):
    """Uses an LLM to create a coherent video script from the clips."""
    generator = pipeline('text2text-generation', model=NARRATIVE_MODEL)
    prompt = f"""
    You are an expert documentary film editor. Your task is to create a short, coherent video script that answers the question: "{query}"
    You have been given a collection of video clips. Select the best clips, reorder them to create a compelling narrative, and present the final script as a JSON array.
    RULES:
    1. The narrative must flow logically.
    2. Discard any redundant or irrelevant clips.
    3. The output MUST be a valid JSON array of objects. Each object must have "start", "end", and "narrative_reason" keys.
    Here are the available clips:
    {json.dumps(relevant_segments, indent=2)}
    Now, generate the final JSON script for the supercut:
    """
    NUM_RE  = r'-?\d+(?:\.\d+)?'
    response = generator(prompt, max_length=1024, num_return_sequences=1, clean_up_tokenization_spaces=True)
    json_response_str = response[0]['generated_text']
    try:
        match = re.search(r'\[.*\]', json_response_str, re.DOTALL)
        if not match:
            # also try escaped
            json_response_str = json_response_str.replace(r'\[', '[').replace(r'\]', ']')
            match = re.search(r'\[.*?\]', json_response_str, re.DOTALL)
        if match:
            fixed = re.sub(
                r'"text"\s*:\s*("(?:(?:\\.)|[^"\\])*")\s*,\s*"start"\s*:\s*(' + NUM_RE + r')\s*,\s*"end"\s*:\s*(' + NUM_RE + r')',
                r'{"text": \1, "start": \2, "end": \3}', match.group(0), flags=re.S)
            fixed = fixed.replace('}{', '},{')  # ensure commas between adjacent objects
            if not fixed.strip().startswith('['):
                fixed = '[' + fixed + ']'
            return json.loads(fixed)
        return None
    except Exception:
        return None

def generate_narration_audio(query, script, job_id):
    """Generates a narration script and synthesizes it to an audio file."""
    generator = pipeline('text2text-generation', model=NARRATIVE_MODEL)
    script_summary = "\n".join([f"- Clip from {s['start']:.1f}s: \"{s['narrative_reason']}\"" for s in script])
    prompt = f"""
    You are a documentary narrator. Your task is to write a compelling voiceover script for a short video that answers the question: "{query}"
    The video consists of the following clips in order:
    {script_summary}
    Write a concise narration. Start with a brief introduction, provide smooth transitions between the clips, and end with a concluding sentence.
    The output should be the narration text ONLY.
    """
    response = generator(prompt, max_length=512, num_return_sequences=1)
    narration_text = response[0]['generated_text']
    
    tts = gTTS(text=narration_text, lang='en')
    audio_path = os.path.join(JOB_DATA_FOLDER, f'{job_id}_narration.mp3')
    tts.save(audio_path)
    return audio_path

def create_supercut(original_video_path, script, query, narration_audio_path=None):
    """Edits the video, optionally adding narration with audio ducking."""
    video = VideoFileClip(original_video_path)
    clips = [video.subclip(float(item['start']), float(item['end'])) for item in script if float(item['end']) <= video.duration]

    if not clips:
        return None

    final_clip = concatenate_videoclips(clips)

    if narration_audio_path:
        narration_audio = AudioFileClip(narration_audio_path)
        final_clip.audio = final_clip.audio.fx(volumex, 0.2)
        final_audio = CompositeAudioClip([final_clip.audio, narration_audio])
        final_clip.audio = final_audio

    title_text = f"Supercut for query:\n'{query}'"
    txt_clip = TextClip(title_text, fontsize=40, color='white', bg_color='black', size=final_clip.size, method='caption').set_duration(3)
    final_with_title = concatenate_videoclips([txt_clip, final_clip])

    sanitized_query = sanitize_filename(query)[:50]
    final_filename = f"{sanitized_query}_{uuid.uuid4().hex[:8]}.mp4"
    output_path = os.path.join(RESULT_FOLDER, final_filename)
    
    final_with_title.write_videofile(output_path, codec="libx264", audio_codec="aac")
    
    video.close()
    return final_filename

# --- Main Pipeline & Q&A Functions ---

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

def process_video_pipeline(youtube_url, query, job_id, jobs, with_narration):
    """The main end-to-end function that orchestrates all stages."""
    video_path = None
    narration_path = None
    try:
        # 1. Get Transcript using the Smart Pipeline
        video_id = get_video_id(youtube_url)
        if not video_id:
            raise ValueError("Invalid YouTube URL")
        
        segments = None
        try:
            # --- FAST PATH: Try to fetch existing captions ---
            update_job_status(job_id, jobs, 'processing', 10, 'Fetching existing captions (fast path)...')
            transcript_list = ytt.fetch(video_id)
            segments = to_segments(transcript_list)
            print(f"Job {job_id}: Successfully fetched captions.")
        except NoTranscriptFound:
             # --- SLOW PATH: Download and transcribe ---
            print(f"Job {job_id}: No captions found. Falling back to deep analysis.")
            update_job_status(job_id, jobs, 'processing', 20, 'No captions found. Downloading video...')
            video_path, _ = download_video(youtube_url, job_id)
            update_job_status(job_id, jobs, 'processing', 40, 'Transcribing audio with Whisper...')
            segments = transcribe_video_whisper(video_path, job_id)

        # Save segments for Q&A bot
        segments_path = os.path.join(JOB_DATA_FOLDER, f'{job_id}_segments.json')
        with open(segments_path, 'w') as f:
            json.dump(segments, f)

        # If we used captions, we still need to download the video for editing
        if video_path is None:
            update_job_status(job_id, jobs, 'processing', 55, 'Captions found. Downloading video for editing...')
            video_path, _ = download_video(youtube_url, job_id)

        # 2. Create Semantic Index
        update_job_status(job_id, jobs, 'processing', 60, 'Creating semantic index...')
        index, model, _, valid_indices = create_semantic_index(segments, job_id)
        if index is None: raise ValueError("Could not process video transcript.")

        # 3. Search for Relevant Clips
        update_job_status(job_id, jobs, 'processing', 65, 'Searching for relevant clips...')
        relevant_segments = search_index(query, index, model, segments, valid_indices)
        if not relevant_segments: raise ValueError("No relevant segments found.")

        # 4. Generate Narrative Script
        update_job_status(job_id, jobs, 'processing', 70, 'Generating narrative script...')
        script = generate_narrative_script(query, relevant_segments)
        if not script: raise ValueError("Narrative Engine failed to produce a script.")

        # 5. Handle Narration (Optional)
        narration_path = None
        if with_narration:
            update_job_status(job_id, jobs, 'processing', 75, 'Generating AI narration...')
            narration_path = generate_narration_audio(query, script, job_id)

        # 6. Create Supercut
        update_job_status(job_id, jobs, 'processing', 85, 'Rendering final video...')
        output_filename = create_supercut(video_path, script, query, narration_path)
        if not output_filename: raise ValueError("Failed to create the final video.")

        # 7. Finalize
        jobs[job_id] = {'status': 'completed', 'progress': 100, 'message': 'Processing complete!', 'filename': output_filename}

    except Exception as e:
        print(f"Error in job {job_id}: {e}")
        jobs[job_id] = {'status': 'failed', 'progress': 100, 'message': str(e)}
    finally:
        if video_path and os.path.exists(video_path): os.remove(video_path)
        if narration_path and os.path.exists(narration_path): os.remove(narration_path)

def answer_question_from_video(question, job_id):
    """Answers a user's question using the pre-processed video data."""
    index_path = os.path.join(JOB_DATA_FOLDER, f'{job_id}.index')
    segments_path = os.path.join(JOB_DATA_FOLDER, f'{job_id}_segments.json')

    if not os.path.exists(index_path) or not os.path.exists(segments_path):
        raise FileNotFoundError("Processed video data not found.")

    index = faiss.read_index(index_path)
    with open(segments_path, 'r') as f:
        segments = json.load(f)

    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Re-create the valid_indices mapping to find original segment from index result
    sentences = [segment['text'] for segment in segments]
    valid_indices = [i for i, s in enumerate(sentences) if s.strip()]

    query_embedding = model.encode([question]).astype('float32')
    _, indices = index.search(query_embedding, k=5)

    context = ""
    for i in indices[0]:
        if i < len(valid_indices):
            original_segment_index = valid_indices[i]
            context += segments[original_segment_index]['text'] + "\n"

    qa_pipeline = pipeline('text2text-generation', model=QA_MODEL)
    prompt = f"""
    Based ONLY on the following context from a video transcript, answer the question.
    If the answer is not in the context, state that the information is not available in the video.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    response = qa_pipeline(prompt, max_length=512)
    return response[0]['generated_text']

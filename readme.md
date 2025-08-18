## 
## 🚀 AI Video Supercut Generator (FastAPI v1.2 - UI Update) 🚀
##
## This project is a complete codebase for the AI Video Supercut Generator,
## now featuring a fully redesigned, modern user interface.
##
## VERSION UPGRADE:
## - UI OVERHAUL: Replaced the basic HTML and CSS with a professional, dark-themed
##   design based on the provided UI mockup. This includes new layouts, color schemes,
##   gradient effects, and custom icons.
## - HTML STRUCTURE: Refactored `index.html` and `result.html` to support the new design.
## - CSS REWRITE: Completely rewrote `style.css` to implement the new aesthetic.
##
## ==============================================================================
## FILE: README.md
## ==============================================================================

# AI Video Supercut Generator (FastAPI v1.2 - UI Update)

This project provides a complete, extensive codebase for an AI-powered application that generates "supercuts" from long-form videos based on a user's natural language query. This version features a major user interface overhaul to match a modern, professional design aesthetic.

### Core Architecture (FastAPI)

The system is designed as a Python FastAPI application, which is run by a Uvicorn ASGI server.

1.  **Asynchronous API:** All web routes are now asynchronous, allowing for higher concurrency and better performance under load.
2.  **Background Tasks:** The long-running video processing pipeline is now handled by FastAPI's `BackgroundTasks`.
3.  **Transcript Retrieval (Smart Pipeline)**: The system first attempts to fetch existing YouTube captions (fast path) and falls back to Whisper transcription (deep analysis path) if needed.
4.  **Narrative Engine (Gemma Powered)**: The backend uses a Gemma model to generate a "playlist" of timestamps based on the user's query and the video transcript.
5.  **Frontend Video Synthesis (YouTube IFrame API)**: The frontend receives the timestamp playlist and uses the YouTube IFrame Player API to create a seamless, stitched-together video experience directly in the user's browser.

### How to Run This Project

**1. Clone the Repository:**
```bash
git clone <repository_url>
cd ai-video-supercut-generator
```

**2. Set Up a Python Environment & Install Dependencies:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```
*Note: `ffmpeg` is still required for the fallback Whisper transcription.*

**3. Run the Application:**
```bash
uvicorn main:app --reload
```
Navigate to `http://127.0.0.1:8000` in your web browser.

### Project Structure

```
/
|-- main.py                 # Main FastAPI application.
|-- video_processor.py      # Core logic for transcript retrieval, semantic search, etc.
|-- requirements.txt        # Python dependencies.
|-- static/
|   |-- script.js           # Frontend JavaScript.
|   |-- style.css           # NEW: Rewritten CSS for the modern UI.
|-- templates/
|   |-- index.html          # NEW: Refactored for the new design.
|   |-- result.html         # NEW: Refactored for the new design.
|-- job_data/               # Stores transcripts and indexes.
|-- uploads/                # Temporary storage for videos needing transcription.
|-- README.md               # This file.
```

---
### Version History

This section documents the major architectural and feature milestones of the project.

**v1.0 (Flask): Core Supercut Generator**
* **Architecture:** Initial version built on the Flask web framework. Used a backend-heavy approach where the server performed all tasks.
* **Workflow:**
    1.  Downloaded the full video file using `yt-dlp`.
    2.  Transcribed audio using `Whisper`.
    3.  Performed semantic search on the transcript to find relevant clips.
    4.  Used a `Flan-T5` LLM as a "Narrative Engine" to create a script.
    5.  Rendered a new MP4 video file on the server using `MoviePy`.
* **Key Challenge:** The process was very slow and resource-intensive due to the full video download and rendering steps.

**v2.0 (Flask): AI Narration & Q&A Bot**
* **New Features:**
    * **AI Narration:** Added an optional feature to generate a voiceover script with an LLM and synthesize it into audio using gTTS. The final video included this narration with audio ducking.
    * **Q&A Bot:** Implemented a Retrieval-Augmented Generation (RAG) bot. After processing, users could ask questions about the video. The system would retrieve relevant text chunks and use an LLM to generate an answer.
* **Architecture:** Still based on the backend-heavy V1 architecture. Introduced a `job_data` directory to persist transcripts and indexes for the Q&A bot.

**v3.0 (Flask): Optimized "Smart Pipeline"**
* **Core Improvement:** Addressed the major performance bottleneck of the previous versions.
* **Workflow Change:**
    * **Fast Path:** The system now first attempts to fetch pre-existing YouTube captions, which is nearly instantaneous.
    * **Fallback Path:** Only if captions are not available does the system fall back to the slow process of downloading the video and running `Whisper`.
* **Impact:** Dramatically improved the speed and user experience for the majority of videos.

**v4.0 (Flask): Advanced Frontend Architecture**
* **Major Architectural Shift:** Decoupled the video "stitching" from the backend to make the application faster and more scalable.
* **New Workflow:**
    1.  The backend's only job is to produce a "playlist" of timestamps. **It no longer downloads or renders any video files.**
    2.  The frontend receives this playlist and uses the **YouTube IFrame Player API** to play the segments directly from YouTube.
    3.  JavaScript handles playing one clip and pre-buffering the next, creating a seamless viewing experience.
* **Other Upgrades:**
    * **Model Upgrade:** Migrated from `Flan-T5` to `Gemma` for better narrative and Q&A quality.
    * **Sentence-Aware Chunking:** Added logic to ensure all clips correspond to complete sentences for better narrative flow.

**v1.1 (FastAPI): Framework Migration**
* **Core Improvement:** The entire application was refactored from Flask to FastAPI.
* **Benefits:**
    * **Performance:** Gained significant speed improvements from FastAPI's asynchronous capabilities.
    * **Modernization:** Replaced multiprocessing with FastAPI's built-in `BackgroundTasks` for cleaner background job handling.
    * **Developer Experience:** Gained features like automatic data validation and API documentation.

**v1.2 (FastAPI): UI Overhaul**
* **Core Improvement:** Replaced the entire frontend with a professional, dark-themed UI based on a user-provided design mockup.
* **Changes:**
    * Rewrote `static/style.css` from scratch to implement the new color palette, gradients, and layout.
    * Restructured `templates/index.html` and `templates/result.html` with new HTML elements and classes to match the design.
    * Added inline SVG icons for a cleaner look and faster loading.
* **Current Version:** This represents the most modern, efficient, and visually polished version of the application.

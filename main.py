import os
from typing import Optional
import uuid
import json
from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

# Import the core processing functions
from video_processor import process_transcript_pipeline, answer_question_from_video

# --- App Initialization ---
app = FastAPI()
load_dotenv()  # Load environment variables from .env file

# --- Static Files and Templates Configuration ---
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/job_data", StaticFiles(directory="job_data"), name="job_data") # Serve narration files
templates = Jinja2Templates(directory="templates")

# --- Create Directories ---
for folder in ['uploads', 'job_data']:
    if not os.path.exists(folder):
        os.makedirs(folder)

# --- In-memory Job Store ---
jobs = {}

# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main homepage."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def create_processing_job(
    background_tasks: BackgroundTasks,
    youtube_url: str = Form(...),
    query: str = Form(...),
    with_narration: bool = Form(False),
):
    """
    Accepts a new job request, assigns a job ID, and starts the
    processing in a background task.
    """
    if not youtube_url or not query:
        return JSONResponse(status_code=400, content={'error': 'YouTube URL and query are required.'})

    job_id = str(uuid.uuid4())
    jobs[job_id] = {'status': 'queued', 'progress': 0, 'message': 'Job is queued...'}

    background_tasks.add_task(process_transcript_pipeline, youtube_url, query, job_id, jobs, with_narration)

    return JSONResponse(content={'job_id': job_id})

@app.post("/qa")
async def handle_qa(job_id: str = Form(...), question: str = Form(...)):
    """Handles a question for the Q&A bot."""
    job = jobs.get(job_id)
    if not job or job.get('status') != 'completed':
        return JSONResponse(status_code=400, content={'error': 'Video processing is not complete.'})

    try:
        answer = answer_question_from_video(question, job_id)
        return JSONResponse(content={'answer': answer})
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Allows the frontend to poll for the status of a job."""
    job = jobs.get(job_id)
    if job:
        return JSONResponse(content=job)
    return JSONResponse(status_code=404, content={'error': 'Job not found'})

@app.get("/result/{job_id}", response_class=HTMLResponse)
async def get_result_page(request: Request, job_id: str):
    """Renders the page to display the final video player."""
    job = jobs.get(job_id)
    if job and job.get('status') == 'completed':
        return templates.TemplateResponse("result.html", {
            "request": request,
            "result_data": json.dumps(job.get('result')),
            "job_id": job_id
        })
    elif job:
        return HTMLResponse(content="Job is still processing or has failed.", status_code=400)
    return HTMLResponse(content="Job not found.", status_code=404)

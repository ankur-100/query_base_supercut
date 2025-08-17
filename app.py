import os
from flask import Flask, request, render_template, jsonify, send_from_directory
import uuid
from multiprocessing import Process, Manager

import debugpy


# Import the core processing functions
from video_processor import process_video_pipeline, answer_question_from_video

app = Flask(__name__)

# Create directories if they don't exist
for folder in ['uploads', 'results', 'job_data']:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Use a Manager dictionary to share job status between processes
manager = Manager()
jobs = manager.dict()

@app.route('/')
def index():
    """Renders the main page with the submission form."""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    """
    Handles the form submission.
    Kicks off the video processing pipeline in a separate process.
    """
    youtube_url = request.form['youtube_url']
    query = request.form['query']
    with_narration = 'with_narration' in request.form
    
    if not youtube_url or not query:
        return jsonify({'error': 'YouTube URL and query are required.'}), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = {'status': 'queued', 'progress': 0, 'message': 'Job is queued...'}

    # Run the heavy processing in a background process
    process = Process(target=process_video_pipeline, args=(youtube_url, query, job_id, jobs, with_narration))
    process.start()

    return jsonify({'job_id': job_id})

@app.route('/qa', methods=['POST'])
def qa():
    """
    Handles a question for the Q&A bot.
    """
    print("CT:", request.headers.get("Content-Type"))
    print("RAW:", request.get_data(as_text=True)[:200])
    print("FORM:", dict(request.form))
    print("JSON:", request.get_json(silent=True))
    job_id = request.form['job_id']
    question = request.form['question']

    if not job_id or not question:
        return jsonify({'error': 'Job ID and question are required.'}), 400
    
    job = jobs.get(job_id)
    if not job or job['status'] != 'completed':
        return jsonify({'error': 'Video processing is not complete for this job.'}), 400

    try:
        answer = answer_question_from_video(question, job_id)
        return jsonify({'answer': answer})
    except Exception as e:
        print(f"Q&A Error for job {job_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status/<job_id>')
def status(job_id):
    """
    Allows the frontend to poll for the status of a job.
    """
    job = jobs.get(job_id)
    if job:
        return jsonify(job)
    return jsonify({'error': 'Job not found'}), 404

@app.route('/result/<job_id>')
def result(job_id):
    """
    Renders the page to display the final generated video.
    """
    job = jobs.get(job_id)
    if job and job['status'] == 'completed':
        return render_template('result.html', filename=job['filename'], job_id=job_id)
    elif job:
        return "Job is still processing or has failed.", 400
    return "Job not found.", 404

@app.route('/results/<filename>')
def serve_video(filename):
    """
    Serves the final video file from the results directory.
    """
    return send_from_directory('results', filename)

if __name__ == '__main__':
    app.run(debug=True)


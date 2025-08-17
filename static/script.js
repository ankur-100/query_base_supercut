
document.addEventListener('DOMContentLoaded', () => {
    // Handler for the main processing form
    const processForm = document.getElementById('process-form');
    if (processForm) {
        processForm.addEventListener('submit', handleProcessFormSubmit);
    }

    // Handler for the Q&A form on the result page
    const qaForm = document.getElementById('qa-form');
    if (qaForm) {
        qaForm.addEventListener('submit', handleQaFormSubmit);
    }
});

function handleProcessFormSubmit(event) {
    event.preventDefault();

    const formData = new FormData(event.target);
    
    document.getElementById('result-container').style.display = 'none';
    document.getElementById('error-container').style.display = 'none';
    document.getElementById('status-container').style.display = 'block';
    updateProgress(0, 'Submitting job...');

    fetch('/process', {
        method: 'POST',
        body: new URLSearchParams(formData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.job_id) {
            pollStatus(data.job_id);
        } else {
            showError(data.error || 'Failed to start job.');
        }
    })
    .catch(error => {
        showError('An unexpected error occurred.');
        console.error('Error:', error);
    });
}

function handleQaFormSubmit(event) {
    event.preventDefault();
    const question = document.getElementById('qa-question').value;
    const jobId = document.getElementById('job_id').value;
    const answerBox = document.getElementById('qa-answer-box');
    const answerP = document.getElementById('qa-answer');
    const spinner = document.getElementById('qa-spinner');
    
    spinner.style.display = 'block';
    answerBox.style.display = 'none';

    fetch('/qa', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8' },
        body: `job_id=${encodeURIComponent(jobId)}&question=${encodeURIComponent(question)}`
    })
    .then(response => response.json())
    .then(data => {
        spinner.style.display = 'none';
        if (data.answer) {
            answerP.textContent = data.answer;
            answerBox.style.display = 'block';
        } else {
            answerP.textContent = data.error || 'Failed to get an answer.';
            answerBox.style.display = 'block';
        }
    })
    .catch(error => {
        spinner.style.display = 'none';
        answerP.textContent = 'An error occurred while fetching the answer.';
        answerBox.style.display = 'block';
        console.error('Q&A Error:', error);
    });
}


function pollStatus(jobId) {
    const interval = setInterval(() => {
        fetch(`/status/${jobId}`)
        .then(response => response.json())
        .then(data => {
            if (data.status === 'completed') {
                clearInterval(interval);
                // Redirect to the result page instead of just showing a link
                window.location.href = `/result/${jobId}`;
            } else if (data.status === 'failed') {
                clearInterval(interval);
                showError(data.message);
            } else {
                updateProgress(data.progress, data.message);
            }
        })
        .catch(error => {
            clearInterval(interval);
            showError('Failed to get job status.');
            console.error('Error:', error);
        });
    }, 3000);
}

function updateProgress(progress, message) {
    document.getElementById('progress-bar-inner').style.width = `${progress}%`;
    document.getElementById('status-message').textContent = message;
}

function showError(message) {
    document.getElementById('status-container').style.display = 'none';
    const errorContainer = document.getElementById('error-container');
    document.getElementById('error-message').textContent = message;
    errorContainer.style.display = 'block';
}

// --- Main Form and Polling Logic ---

document.addEventListener('DOMContentLoaded', () => {
    const processForm = document.getElementById('process-form');
    if (processForm) {
        processForm.addEventListener('submit', handleProcessFormSubmit);
    }

    const qaForm = document.getElementById('qa-form');
    if (qaForm) {
        qaForm.addEventListener('submit', handleQaFormSubmit);
    }
});

function handleProcessFormSubmit(event) {
    event.preventDefault();
    const formData = new FormData(event.target);
    
    document.getElementById('error-container').style.display = 'none';
    const statusContainer = document.getElementById('status-container');
    statusContainer.style.display = 'block';
    document.querySelector('#process-form').style.display = 'none';
    document.querySelector('.features').style.display = 'none';

    updateProgress(0, 'Submitting job...');

    fetch('/process', {
        method: 'POST',
        body: new URLSearchParams(formData)
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.detail || 'Failed to start job.'); });
        }
        return response.json();
    })
    .then(data => {
        if (data.job_id) {
            pollStatus(data.job_id);
        } else {
            showError(data.error || 'Failed to start job.');
        }
    })
    .catch(error => showError(error.message));
}

function pollStatus(jobId) {
    const interval = setInterval(() => {
        fetch(`/status/${jobId}`)
        .then(response => response.json())
        .then(data => {
            if (data.status === 'completed') {
                clearInterval(interval);
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
        });
    }, 3000);
}

// --- YouTube Player Logic (result.html) ---

let player;
let playlist = [];
let currentSegmentIndex = 0;
let seekTimeout;

// This function is called automatically by the YouTube API script once it's ready.
function onYouTubeIframeAPIReady() {
    if (typeof resultData !== 'undefined' && resultData) {
        initializePlayer(resultData);
    }
}

function initializePlayer(data) {
    playlist = data.playlist;
    const videoId = data.video_id;

    if (!playlist || playlist.length === 0) {
        document.getElementById('narrative-display').innerHTML = "<p>Narrative engine could not generate a playlist.</p>";
        return;
    }

    // Handle Narration Player
    if (data.narration_url) {
        const narrationContainer = document.getElementById('narration-player-container');
        const narrationPlayer = document.getElementById('narration-player');
        narrationPlayer.src = data.narration_url;
        narrationContainer.style.display = 'block';
    }

    const playlistUl = document.getElementById('playlist-list');
    playlist.forEach((item, index) => {
        const li = document.createElement('li');
        li.innerHTML = `<span>${item.speaker || 'SEGMENT ' + (index + 1)}</span>${item.narrative_reason}`;
        li.onclick = () => jumpToSegment(index);
        playlistUl.appendChild(li);
    });

    player = new YT.Player('player', {
        height: '390',
        width: '640',
        videoId: videoId,
        playerVars: { 'playsinline': 1, 'autoplay': 1, 'controls': 1, 'rel': 0, 'modestbranding': 1 },
        events: { 'onReady': onPlayerReady, 'onStateChange': onPlayerStateChange }
    });
}

function onPlayerReady(event) {
    jumpToSegment(0);
}

function onPlayerStateChange(event) {
    clearTimeout(seekTimeout);
    if (event.data === YT.PlayerState.PLAYING) {
        const currentSegment = playlist[currentSegmentIndex];
        const checkTime = () => {
            if (player.getCurrentTime() >= currentSegment.end) {
                playNextSegment();
            } else {
                seekTimeout = setTimeout(checkTime, 250);
            }
        };
        checkTime();
    }
}

function jumpToSegment(index) {
    if (index >= 0 && index < playlist.length) {
        currentSegmentIndex = index;
        const segment = playlist[index];
        player.seekTo(segment.start, true);
        player.playVideo();
        updatePlaylistUI();
    }
}

function playNextSegment() {
    if (currentSegmentIndex < playlist.length - 1) {
        jumpToSegment(currentSegmentIndex + 1);
    } else {
        player.stopVideo();
        updatePlaylistUI(true); 
    }
}

function updatePlaylistUI(finished = false) {
    const items = document.querySelectorAll('#playlist-list li');
    items.forEach((item, index) => {
        item.classList.remove('active');
        if (!finished && index === currentSegmentIndex) {
            item.classList.add('active');
        }
    });
}


// --- Q&A Bot and UI Helper Logic ---

function handleQaFormSubmit(event) {
    event.preventDefault();
    const questionInput = document.getElementById('qa-question');
    const question = questionInput.value;
    const jobId = document.getElementById('job_id').value;
    const answerBox = document.getElementById('qa-answer-box');
    const answerP = document.getElementById('qa-answer');
    const spinner = document.getElementById('qa-spinner');

    spinner.style.display = 'block';
    answerBox.style.display = 'none';

    fetch('/qa', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: `job_id=${encodeURIComponent(jobId)}&question=${encodeURIComponent(question)}`
    })
    .then(response => response.json())
    .then(data => {
        spinner.style.display = 'none';
        answerP.textContent = data.answer || data.error || 'Failed to get an answer.';
        answerBox.style.display = 'block';
        questionInput.value = '';
    })
    .catch(error => {
        spinner.style.display = 'none';
        answerP.textContent = 'An error occurred while fetching the answer.';
        answerBox.style.display = 'block';
    });
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
    document.querySelector('#process-form').style.display = 'flex';
    document.querySelector('.features').style.display = 'grid';
}
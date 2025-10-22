# 🚀 AI Video Supercut Generator

This document provides a  overview of the **AI Video Supercut Generator**, an AI-powered application that generates “supercuts” from long-form videos based on a user's natural language query.  
It details the current architecture, the strategic roadmap for future enhancements, and the project's version history.

---

## 1. Overview

The **AI Video Supercut Generator** is a state-of-the-art platform designed to address information overload in long-form video content.  
By transforming the passive, linear viewing experience into an active, on-demand knowledge retrieval process, it allows users to extract precise information from hours-long videos in minutes.

The system leverages a sophisticated multimodal AI pipeline to ingest, analyze, and re-sequence video content, delivering a concise, narratively coherent "supercut" that directly answers a user's query.  
This makes deep knowledge more accessible and maximizes the user's return on time invested.

---

## 2. System Architecture & Future Roadmap

The platform is built on a modular, multi-stage pipeline designed for performance, scalability, and continuous improvement.

### 2.1 Ingestion and Caching

When a user submits a video, the system first checks a caching layer.  
If the video has been processed before, the existing transcript and metadata are retrieved instantly.

**[Implemented] Transcript Caching:**  
A caching layer is in place to store transcripts from previously processed videos.  
This is a critical feature that reduces redundant processing, saving both money on ASR services and time for the end-user, dramatically improving the experience for popular content.

---

### 2.2 Multimodal Analysis Pipeline

If a video is new, it enters the full analysis pipeline, which understands content from both audio and visual streams.

#### Audio Analysis (ASR)

The audio track is processed to generate a highly accurate, timestamped transcript with speaker labels.

**Current (v2.0 “Audio-First” Pipeline):**  
- Downloads the audio track  
- Processes via `pyannote.audio` for speaker diarization  
- Uses `WhisperX` for word-level timestamps aligned with speaker timeline  

**Planned Enhancements:**
- **Latency Reduction:** Parallelization with `FFmpeg` to chunk audio intelligently for concurrent processing across CPU cores.  
- **Cost Optimization:** Evaluate migration to managed ASR APIs (e.g., **Deepgram**) or fully self-hosted **faster-whisper** setups for cost control and robustness.

---

#### Visual Analysis (VLM & CV)

This major planned feature extracts textual and contextual information from video frames.

**Planned Strategy:**  
A computationally efficient two-stage, “coarse-to-fine” pipeline:  
1. Lightweight scan flags likely text-containing frames.  
2. Candidate frames are passed to a deeper analysis pipeline.

**Planned Implementation:**
- **CogVLM:** For OCR and vision-language reasoning, chosen for strong VQA/OCR performance.  
- **RF-DETR:** For object detection, offering high accuracy and speed without post-processing (NMS).

---

### 2.3 Data Indexing

All extracted data (transcripts, OCR text, object labels, etc.) are converted into vector embeddings and stored in a high-performance vector database.

- **Current:** FAISS  
- **Planned Upgrade:** Migrate to **Qdrant** or **Milvus** for scalability, filtering, and GPU-accelerated workloads.

---

### 2.4 Narrative Synthesis Engine & Query Routing

This is the core “brain” that constructs the final supercut.

**Current (v2.0):**
- Performs semantic search for relevant clips  
- LLM generates a `narrative_reason` for each  
- Uses a Knapsack algorithm to maximize **relevance + diversity** within user time budget  

**Planned Enhancements:**
- **Latency Reduction:** Replace multiple API calls with a **self-hosted fine-tuned LLM**.  
- **Model:** Llama 3 or Qwen  
- **Fine-tuning:** QLoRA for efficient training  
- **Inference:** Deploy via **vLLM** or **Text Generation Inference (TGI)**  

**Intelligent Query Routing (Future MoE Setup):**
- **DIVERSITY_QUERY:** Use Knapsack algorithm for broad queries.  
- **DENSITY_QUERY:** Retrieve dense semantic clusters for specific, detailed answers.

---

### 2.5 User Management & Deployment

**Planned Feature (User Authentication):**  
To enable personalization and tracking.

- **Initial Implementation:** Firebase Authentication (free up to 50K MAUs)  
- **Database:** MySQL for user preferences (secured, hashed credentials)  

**Planned Deployment:**  
Deployed on **AWS (Singapore region)** with:  
- Multi-AZ architecture  
- SageMaker for model endpoints  
- CloudFront for global delivery  

---

## 3. Technology Stack

| Category | Current | Planned Enhancements & Alternatives |
|-----------|----------|------------------------------------|
| **Framework** | FastAPI + Uvicorn | — |
| **Cloud & Deployment** | Local | AWS (SageMaker, CloudFront, S3, EC2) |
| **User Authentication** | None | Firebase, MySQL backend |
| **Databases** | FAISS, Local | Qdrant / Milvus, Managed MySQL |
| **Audio Processing** | WhisperX, pyannote.audio | Deepgram API, faster-whisper |
| **Language Models** | External (Gemini) | Self-hosted Llama 3 / Qwen + QLoRA |
| **Inference Servers** | None | vLLM / TGI |
| **Vision Models** | None | CogVLM (OCR), RF-DETR (Object Detection) |

---

## 4. How to Run This Project

### 1️⃣ Clone the Repository

```bash
git clone <repository_url>
cd ai-video-supercut-generator

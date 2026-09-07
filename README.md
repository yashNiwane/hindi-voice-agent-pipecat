# Production Real-Time Hindi Voice Telecalling Agent

A production-oriented, ultra-low latency real-time Hindi voice telecalling agent built using **Pipecat**, **WebRTC**, and open-weight models optimized for **Kaggle Tesla T4 GPU (16GB VRAM)**.

---

## 1. Verified Models & Hugging Face Identifiers

| Layer | Architecture / Model | Hugging Face Repository | Runtime Engine | VRAM (T4) |
|---|---|---|---|---|
| **ASR** | Qwen3-ASR (0.6B) | [`Qwen/Qwen3-ASR-0.6B-hf`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) | PyTorch / Transformers | ~1.3 GB |
| **LLM** | Gemma 4 E2B Q4_0 | [`google/gemma-4-E2B-it-qat-q4_0-gguf`](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf) (`gemma-4-E2B_q4_0-it.gguf`) | `llama-cpp-python` (GPU offload) | ~1.8 GB |
| **TTS** | MMS-TTS Hindi (VITS 16kHz) | [`facebook/mms-tts-hin`](https://huggingface.co/facebook/mms-tts-hin) | PyTorch / Transformers | ~0.5 GB |
| **Noise Reduction** | RNNoise recurrent neural filter | `pyrnnoise` via `pipecat-ai[rnnoise]` | C++ RNN library | ~10 MB |
| **Voice Activity Detection** | Silero VAD | `pipecat-ai[silero]` | PyTorch / ONNX | ~50 MB |
| **Audio Transport** | SmallWebRTC | `pipecat-ai[webrtc,runner]` + `aiortc` | AsyncIO / WebRTC | Minimal |
| **Total Memory** | **Complete local stack** | **100% Open Weights / No Paid APIs** | | **~3.7 GB / 15.0 GB** |

---

## 2. Pipeline Flow

```text
Laptop Browser (Mic)
       │
       │ WebRTC (16kHz PCM / Opus)
       ▼
Cloudflare Tunnel (Public HTTPS URL)
       │
       ▼
Kaggle Server (FastAPI on Port 7860)
       │
       ├── Pipecat SmallWebRTC Input Transport
       ├── RNNoise (Real-time neural noise cancellation)
       ├── Silero VAD (0.15s barge-in / 0.5s turn-taking pause)
       ├── Qwen3-ASR (Hindi speech-to-text)
       ├── Gemma 4 E2B Q4 (Streaming token generation on GPU)
       ├── MMS-TTS Hindi (Chunked 20ms audio frame synthesis)
       └── Pipecat SmallWebRTC Output Transport
       │
       │ WebRTC (Downstream audio)
       ▼
Laptop Browser (Speaker)
```

---

## 3. Project Structure

- `config/settings.py`: Centralized configuration, VAD thresholds, model IDs, system prompt.
- `modules/asr_service.py`: Qwen3-ASR STT service integrated with Pipecat pipeline.
- `modules/llm_service.py`: Gemma 4 E2B Q4 GGUF LLM service with token-by-token streaming.
- `modules/tts_service.py`: MMS-TTS Hindi TTSService chunking into 20ms WebRTC frames.
- `modules/vad_noise.py`: Silero VAD & RNNoise neural noise suppression filter.
- `modules/latency_tracker.py`: Real-time instrumentation (TTFT, TTFA, E2E turn latency).
- `tests/test_asr.py`: Standalone ASR verification script (Phase 2).
- `tests/test_llm.py`: Standalone LLM streaming test (Phase 3).
- `tests/test_tts.py`: Standalone Hindi TTS benchmark (Phase 4).
- `tests/test_pipeline.py`: Chained ASR -> LLM -> TTS pipeline test (Phase 5).
- `static/index.html`: Modern WebRTC in-browser calling interface with audio visualizer.
- `bot.py`: Pipecat pipeline definition with turn-taking and barge-in handling.
- `server.py`: FastAPI server handling WebRTC signaling (`/api/offer`) and serving client UI.
- `start_kaggle.sh`: Automation script to launch Cloudflare Tunnel and start server.
- `kaggle_deployment.ipynb`: Interactive Kaggle notebook for 1-click execution.
- `requirements.txt`: Pinned dependencies.

---

## 4. Kaggle Deployment Instructions

1. Start a **New Notebook** on [Kaggle](https://www.kaggle.com/).
2. In the right panel settings:
   - **Accelerator**: Choose **GPU T4 x1**.
   - **Internet**: Switch to **On**.
3. Run the cells in `kaggle_deployment.ipynb` or execute in a terminal:
   ```bash
   pip install --quiet --upgrade pip
   pip install --quiet "pipecat-ai[silero,rnnoise,webrtc,runner]>=0.0.50"
   pip install --quiet aiortc fastapi uvicorn python-dotenv pydantic loguru soundfile soxr pyrnnoise transformers accelerate huggingface_hub
   CMAKE_ARGS="-DGGML_CUDA=on" pip install --quiet llama-cpp-python
   bash start_kaggle.sh
   ```
4. Copy the generated `https://*.trycloudflare.com` URL.
5. Open that URL on your Windows laptop browser, grant microphone access, and click **Start Hindi Call**.

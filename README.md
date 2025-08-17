# Audio-to-Text RAG for Podcast Search

A multimodal Retrieval-Augmented Generation (RAG) system for searching topics across podcast episodes with timestamped references. It converts audio → text, performs (optional) lightweight speaker diarization, indexes chunks in a vector database, and returns contextual answers with precise timestamps.

## ✨ Features
- **Audio → Text** using `faster-whisper` (local) with word-level timestamps.
- **(Optional) Speaker diarization** via MFCC + clustering (no external tokens), yielding anonymous speakers (e.g., `SPEAKER_0`, `SPEAKER_1`). You can swap in `pyannote.audio` if you have a HF token.
- **Chunking** tuned for speech: merges adjacent segments up to a target size with overlap.
- **Vector search** with ChromaDB + Sentence Transformers embeddings.
- **Multi-episode search** and timestamped results.
- **Context-aware synthesis**: either OpenAI (if `OPENAI_API_KEY` is set) or a built‑in extractive composer.
- **Streamlit UI** for ingestion & search. Ready for Hugging Face Spaces deployment.

## 🧱 Architecture
```
Streamlit UI
  ├─ Ingestion: upload audio → preprocess → transcribe → (optional) diarize → chunk → index
  └─ Search: query → embed → retrieve (MMR) → rerank → synthesize → show timestamps
```

---

## 🚀 Quickstart (Local)

> Prereqs: Python 3.10+, FFmpeg (for audio), and optionally a GPU for faster transcription.

1. **Install**:
   ```bash
   python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
   pip install -r requirements.txt
   ```

2. **(Optional) OpenAI key for generation**  
   Create `.env` and add:
   ```bash
   OPENAI_API_KEY=sk-...
   ```

3. **Run the app**:
   ```bash
   streamlit run app.py
   ```

4. **Ingest episodes**: use the sidebar → *Ingest Episodes*. Provide a title and upload audio(s).  
   The system transcribes, (optionally) diarizes, chunks, and indexes into Chroma.

5. **Search**: enter a question/topic. You’ll see top matches across episodes with timestamps and context.  

---

## ☁️ Deploy to Hugging Face Spaces (Streamlit)
1. Create a new **Space** → Template: *Streamlit*.
2. Push this repo to the Space.
3. Optionally add secrets:
   - `OPENAI_API_KEY` (for generative answers)
4. (Spaces CPU works; for speed enable GPU.)

> FFmpeg is required. In Spaces, set **System Packages** to include `ffmpeg` (Settings → Hardware & build → System packages).

---

## 🔧 Configuration
Edit `config.py`:
- `ASR_ENGINE` = `"faster-whisper"` (default) or `"openai"` (requires API key).
- `WHISPER_MODEL_SIZE` = `"small"` (good balance), alternatives: `"tiny"`, `"base"`, `"medium"`.
- `EMBEDDING_MODEL_NAME` = `"sentence-transformers/all-MiniLM-L6-v2"` (fast) or multilingual models.
- `DIARIZATION_MODE` = `"clustering"` or `"none"`.
- Chunk sizes & overlap, top‑k, MMR etc.

---

## 📦 Project Structure
```
audio2text-rag-podcasts/
├── app.py
├── ingest.py
├── search_backend.py
├── config.py
├── requirements.txt
├── utils/
│   ├── __init__.py
│   ├── audio.py
│   ├── chunking.py
│   ├── diarize.py
│   ├── index.py
│   └── eval.py
├── data/
│   ├── raw/
│   └── processed/
├── episodes/
├── chroma_db/
├── examples/
│   └── queries.jsonl
└── README.md
```

---

## 🧪 Evaluation (baseline)
Use `examples/queries.jsonl` to store test queries with expected keywords. Run:
```bash
python -m utils.eval --k 5 --queries examples/queries.jsonl
```
Metrics:
- **Hit@k**, **MRR**, and average **latency** (ms) for retrieval.

> For full RAG quality scoring, integrate **RAGAS** later (optional), but this baseline ships without heavy dependencies.

---

## 🔁 Data Flow & Reproducibility
- All raw audio saved under `data/raw`.
- Intermediate JSON transcript & segments under `data/processed`.
- Vector store persisted in `chroma_db`.

---

## 🧩 Notes on Diarization
The default diarization uses MFCC embeddings + agglomerative clustering to produce anonymous speaker labels, which works well enough for demos. For higher quality, install `pyannote.audio` and modify `utils/diarize.py` to use the pretrained pipeline (requires a HF token).

---

## 🛡️ License
MIT

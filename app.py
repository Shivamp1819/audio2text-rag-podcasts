from my_package.ingest import process_episode
import json
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from config import (DATA_RAW, DATA_PROCESSED, EPISODES_DIR, CHUNK_CHAR_LENGTH, CHUNK_CHAR_OVERLAP,
                    TOP_K, MMR_LAMBDA, MAX_UPLOAD_MB)
from ingest import process_episode
from utils.index import reset_index
from search_backend import query_system

st.set_page_config(page_title="Podcast RAG Search", layout="wide")

st.title("🎙️ Audio → Text RAG for Podcast Search")
st.caption("Search topics across episodes with timestamps. Upload → Index → Ask.")

with st.sidebar:
    st.header("⚙️ Ingest Episodes")
    with st.form("ingest_form"):
        episode_title = st.text_input("Episode title", "Untitled episode")
        files = st.file_uploader("Upload audio (mp3, wav, m4a, mp4)", type=["mp3","wav","m4a","mp4"], accept_multiple_files=True)
        diarize = st.checkbox("Enable diarization (clustering)", value=True)
        submitted = st.form_submit_button("Ingest & Index")


    if st.button("🔄 Reset Vector Index"):
        reset_index()
        st.success("Vector index reset. You can re-ingest episodes now.")

    st.markdown("---")
    st.header("🔍 Search Settings")
    top_k = st.slider("Top-K results", min_value=1, max_value=10, value=TOP_K)
    mmr_lambda = st.slider("MMR diversity", min_value=0.0, max_value=1.0, value=MMR_LAMBDA)

if submitted and files:
    for f in files:
        suffix = Path(f.name).suffix.lower()
        path = DATA_RAW / f.name

        import os
        os.makedirs(DATA_RAW, exist_ok=True)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        with st.spinner(f"Processing {f.name} ..."):
            info = process_episode(episode_title=episode_title, file_path=path, diarize=diarize)
        st.success(f"Indexed {f.name}: {len(info['chunks'])} chunks from {len(info['segments'])} segments.")

st.markdown("---")
st.subheader("Ask a question across episodes")
query = st.text_input("Your query", placeholder="e.g., What did the guest say about blockchain regulation?")


if st.button("Search") and query.strip():
    t0 = time.time()
    res = query_system(query, top_k=top_k, mmr_lambda=mmr_lambda)
    elapsed = (time.time() - t0) * 1000
    st.write(f"Latency: {elapsed:.0f} ms")


    st.markdown("### Answer")
    st.markdown(res["answer"])

    st.markdown("### Top Results")
    for i, r in enumerate(res["results"], start=1):
        start_sec = r['start']
        end_sec = r['end']
        start_min = int(start_sec // 60); start_rem = int(start_sec % 60)
        end_min = int(end_sec // 60); end_rem = int(end_sec % 60)
        ts = f"{start_min:02d}:{start_rem:02d}–{end_min:02d}:{end_rem:02d}"
        with st.expander(f"#{i} {r['episode_title']}  •  {ts}"):
            st.write(r['text'])
            st.json({
                "episode_id": r['episode_id'],
                "speakers": r['speakers'],
                "distance": r['distance'],
                "audio_path": r['audio_path'],
                "start_sec": r['start'],
                "end_sec": r['end'],
            })

    st.download_button("⬇️ Download raw results (JSON)", data=json.dumps(res, ensure_ascii=False, indent=2), file_name="results.json", mime="application/json")

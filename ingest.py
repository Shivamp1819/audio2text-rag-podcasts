import json
import uuid
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
load_dotenv()

from faster_whisper import WhisperModel

from config import DATA_RAW, DATA_PROCESSED, EPISODES_DIR, WHISPER_MODEL_SIZE, DIARIZATION_MODE, CHUNK_CHAR_LENGTH, CHUNK_CHAR_OVERLAP
from utils.audio import ensure_wav_16k_mono
from utils.diarize import diarize_by_clustering
from utils.chunking import merge_segments_to_chunks
from utils.index import ChunkRecord, index_chunks


def transcribe_audio(audio_path: Path) -> Dict:
    model = WhisperModel(WHISPER_MODEL_SIZE, device="auto", compute_type="auto")
    segments, info = model.transcribe(str(audio_path), vad_filter=True, word_timestamps=False)
    out = []
    for i, seg in enumerate(segments):
        out.append({
            "id": i,
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip()
        })
    return {
        "language": info.language,
        "segments": out
    }


def process_episode(episode_title: str, file_path: Path, episode_id: str = None, diarize: bool = True) -> Dict:
    episode_id = episode_id or str(uuid.uuid4())
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    EPISODES_DIR.mkdir(parents=True, exist_ok=True)

    # Save original file
    target = DATA_RAW / f"{episode_id}_{Path(file_path).name}"
    if str(file_path) != str(target):
        with open(file_path, 'rb') as src, open(target, 'wb') as dst:
            dst.write(src.read())

    wav_path = ensure_wav_16k_mono(target)

    # Transcribe
    asr = transcribe_audio(wav_path)
    segments = asr['segments']

    # Diarize (optional)
    if diarize and DIARIZATION_MODE == "clustering" and len(segments) >= 2:
        speakers = diarize_by_clustering(str(wav_path), segments)
        for s, seg in zip(speakers, segments):
            seg['speaker'] = s
    else:
        for seg in segments:
            seg['speaker'] = 'UNKNOWN'

    # Chunk
    chunks = merge_segments_to_chunks(segments, max_chars=CHUNK_CHAR_LENGTH, overlap=CHUNK_CHAR_OVERLAP)

    # Persist transcript JSON
    transcript_json = {
        "episode_id": episode_id,
        "episode_title": episode_title,
        "audio_path": str(target),
        "language": asr['language'],
        "segments": segments,
        "chunks": chunks,
    }
    out_json_path = DATA_PROCESSED / f"{episode_id}.json"
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(transcript_json, f, ensure_ascii=False, indent=2)

    # Index chunks
    records = []
    for ch in chunks:
        records.append(
            ChunkRecord(
                id=f"{episode_id}:{ch['chunk_id']}",
                episode_id=episode_id,
                episode_title=episode_title,
                audio_path=str(target),
                start=float(ch['start']),
                end=float(ch['end']),
                text=ch['text'],
                speakers=", ".join(ch['speakers']) if isinstance(ch['speakers'], list) else str(ch['speakers']),
            )
        )

    index_chunks(records)

    return transcript_json
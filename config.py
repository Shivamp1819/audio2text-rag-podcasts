from pathlib import Path

# --- ASR ---
ASR_ENGINE: str = "faster-whisper"  # "faster-whisper" or "openai"
WHISPER_MODEL_SIZE: str = "small"   # tiny/base/small/medium

# --- Diarization ---
DIARIZATION_MODE: str = "clustering"  # "clustering" or "none"

# --- Embeddings / Retrieval ---
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_DIR: Path = Path("chroma_db")
COLLECTION_NAME: str = "podcasts"
TOP_K: int = 5
MMR_LAMBDA: float = 0.4  # 0..1 diversity

# --- Chunking ---
CHUNK_CHAR_LENGTH: int = 800
CHUNK_CHAR_OVERLAP: int = 120

# --- Paths ---
DATA_RAW: Path = Path("data/raw")
DATA_PROCESSED: Path = Path("data/processed")
EPISODES_DIR: Path = Path("episodes")

# --- UI ---
MAX_UPLOAD_MB: int = 200

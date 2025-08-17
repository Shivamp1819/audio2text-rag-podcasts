from typing import List, Dict, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import time
import uuid
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from config import CHROMA_DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME

@dataclass
class ChunkRecord:
    id: str
    episode_id: str
    episode_title: str
    audio_path: str
    start: float
    end: float
    text: str
    speakers: list

def get_client():
    return chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

def get_or_create_collection(client=None):
    if client is None:
        client = get_client()
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    colls = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in colls:
        return client.get_collection(COLLECTION_NAME, embedding_function=embedder)
    return client.create_collection(COLLECTION_NAME, embedding_function=embedder, metadata={
        "hnsw:space": "cosine"
    })

def index_chunks(chunks: List[ChunkRecord]):
    client = get_client()
    coll = get_or_create_collection(client)
    ids = [c.id for c in chunks]
    texts = [c.text for c in chunks]
    metadatas = []

    for c in chunks:
        # normalize speakers: always store as comma-separated string
        speakers_val = (
            ", ".join(c.speakers) if isinstance(c.speakers, list) else str(c.speakers)
        )

        metadatas.append({
            "episode_id": c.episode_id,
            "episode_title": c.episode_title,
            "audio_path": c.audio_path,
            "start": float(c.start),
            "end": float(c.end),
            "speakers": speakers_val,
        })

    coll.add(ids=ids, documents=texts, metadatas=metadatas)


def reset_index():
    client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    get_or_create_collection(client)

def bm25_rerank(query: str, docs: List[str], top_k: int = 5) -> List[int]:
    tokenized_corpus = [d.split() for d in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.split())
    idxs = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:top_k]
    return idxs

def search(query: str, top_k: int = 5, mmr_lambda: float = 0.4):
    client = get_client()
    coll = get_or_create_collection(client)
    res = coll.query(query_texts=[query], n_results=max(20, top_k*3), include=['documents', 'metadatas', 'distances'])
    if not res['ids'] or not res['ids'][0]:
        return []

    docs = res['documents'][0]
    metas = res['metadatas'][0]
    dists = res['distances'][0]
    candidates = list(range(len(docs)))

    # MMR diversification
    selected = []
    selected_texts = []
    import numpy as np
    lambda_ = mmr_lambda
    while len(selected) < min(top_k, len(candidates)):
        best_i = None
        best_score = -1e9
        for i in candidates:
            relevance = -dists[i]  # smaller distance → higher similarity
            diversity = 0.0
            if selected_texts:
                # cosine similarity proxy by BM25 ranks on text; keep simple
                diversity = max([len(set(docs[i].split()) & set(t.split()))/max(1,len(set(docs[i].split())|set(t.split()))) for t in selected_texts])
            score = lambda_ * relevance - (1 - lambda_) * diversity
            if score > best_score:
                best_score = score
                best_i = i
        selected.append(best_i)
        selected_texts.append(docs[best_i])
        candidates.remove(best_i)

    # BM25 rerank as final tie-breaker
    reranked_idxs = bm25_rerank(query, [docs[i] for i in selected], top_k=len(selected))
    final = [selected[i] for i in reranked_idxs]

    results = []
    for i in final:
        m = metas[i]
        results.append({
            "text": docs[i],
            "episode_id": m["episode_id"],
            "episode_title": m["episode_title"],
            "audio_path": m["audio_path"],
            "start": m["start"],
            "end": m["end"],
            "speakers": m.get("speakers", []),
            "distance": dists[i],
        })
    return results

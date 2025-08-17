import os
import json
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from config import TOP_K, MMR_LAMBDA
from utils.index import search as vs_search

def synthesize_answer(query: str, contexts: List[Dict]) -> str:
    """Compose an answer. If OPENAI_API_KEY is present use GPT; else build extractive answer."""
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        client = OpenAI(api_key=api_key)
        context_text = "\n\n".join([
            f"Episode: {c['episode_title']} [{c['start']:.1f}-{c['end']:.1f}]\n{c['text']}" for c in contexts
        ])
        prompt = f"""You are a helpful assistant answering questions about podcast episodes.
Use the CONTEXT to answer the USER QUESTION concisely with bullet points and include episode title and timestamps in parentheses when referencing facts.
If unsure, say so and suggest the closest matches.

USER QUESTION: {query}

CONTEXT:
{context_text}
"""
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a concise assistant for podcast search."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=400,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            pass  # fall back below

    # Fallback: extractive composer
    bullet_points = []
    for c in contexts:
        summary = c['text']
        if len(summary) > 300:
            summary = summary[:300].rsplit('.', 1)[0] + '.'
        bullet_points.append(f"- {summary} (Episode: {c['episode_title']} @ {c['start']:.0f}s–{c['end']:.0f}s)")
    return "\n".join(bullet_points)

def query_system(query: str, top_k: int = TOP_K, mmr_lambda: float = MMR_LAMBDA) -> Dict:
    results = vs_search(query, top_k=top_k, mmr_lambda=mmr_lambda)
    answer = synthesize_answer(query, results)
    return {"answer": answer, "results": results}

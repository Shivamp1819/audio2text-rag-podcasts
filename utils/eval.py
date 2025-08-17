import time
import json
import argparse
from statistics import mean
from typing import List, Dict
from utils.index import search

def hit_at_k(expected_keywords: List[str], retrieved_texts: List[str]) -> float:
    haystack = " \n ".join(retrieved_texts).lower()
    return float(all(any(kw.lower() in haystack for kw in expected_keywords)))

def mrr(expected_keywords: List[str], retrieved_texts: List[str]) -> float:
    for i, t in enumerate(retrieved_texts, start=1):
        for kw in expected_keywords:
            if kw.lower() in t.lower():
                return 1.0 / i
    return 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queries', required=True, help='Path to queries.jsonl')
    ap.add_argument('--k', type=int, default=5)
    args = ap.parse_args()

    with open(args.queries, 'r', encoding='utf-8') as f:
        queries = [json.loads(line) for line in f]

    m_hits, m_mrr, latencies = [], [], []

    for q in queries:
        qtext = q['query']
        expected = q.get('keywords', [])
        t0 = time.time()
        res = search(qtext, top_k=args.k)
        latencies.append((time.time() - t0) * 1000.0)
        texts = [r['text'] for r in res]
        m_hits.append(hit_at_k(expected, texts))
        m_mrr.append(mrr(expected, texts))

    print(f"Hit@{args.k}: {mean(m_hits):.3f}")
    print(f"MRR: {mean(m_mrr):.3f}")
    print(f"Avg latency (ms): {mean(latencies):.1f}")

if __name__ == '__main__':
    main()

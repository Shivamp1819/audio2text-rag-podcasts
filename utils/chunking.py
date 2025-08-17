from typing import List, Dict

def merge_segments_to_chunks(segments: List[dict], max_chars: int = 800, overlap: int = 120) -> List[dict]:
    """Merge ASR segments into chunks bounded by character length with overlap.
    Each segment is a dict with keys: start, end, text, speaker (optional).
    Returns list of chunk dicts with: start, end, text, speakers(set), segment_ids(list).
    """
    chunks = []
    buf = []
    current_len = 0
    for i, seg in enumerate(segments):
        seg_text = seg.get("text", "").strip()
        if not seg_text:
            continue
        seg_len = len(seg_text) + 1  # space
        if current_len + seg_len > max_chars and buf:
            # flush buffer to chunk
            chunk_text = " ".join([b["text"].strip() for b in buf]).strip()
            chunk = {
                "start": buf[0]["start"],
                "end": buf[-1]["end"],
                "text": chunk_text,
                "speakers": list({b.get("speaker", "UNKNOWN") for b in buf}),
                "segment_ids": [b.get("id", i) for b in buf],
            }
            chunks.append(chunk)

            # create overlap by carrying tail tokens
            carry_text = ""
            carry = []
            carry_len = 0
            for b in reversed(buf):
                t = b["text"].strip()
                if carry_len + len(t) + 1 <= overlap:
                    carry.insert(0, b)
                    carry_len += len(t) + 1
                else:
                    break
            buf = carry[:]
            current_len = sum(len(b["text"]) + 1 for b in buf)

        buf.append(seg)
        current_len += seg_len

    if buf:
        chunk_text = " ".join([b["text"].strip() for b in buf]).strip()
        chunk = {
            "start": buf[0]["start"],
            "end": buf[-1]["end"],
            "text": chunk_text,
            "speakers": list({b.get("speaker", "UNKNOWN") for b in buf}),
            "segment_ids": [b.get("id", 0) for b in buf],
        }
        chunks.append(chunk)

    # assign sequential chunk ids
    for idx, ch in enumerate(chunks):
        ch["chunk_id"] = idx

    return chunks

import math
import os
import re
import uuid
from typing import Dict, List, Set, Tuple
import httpx
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

from app.models.schemas import ExtractedQuestion, PriorityTier

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    SentenceTransformer = None

_MODEL_INSTANCE = None

STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "in", "of", "and", "or", "for", "to",
    "with", "explain", "discuss", "describe", "define", "state", "draw", "write",
    "note", "notes", "short", "detail", "details", "brief", "neat", "diagram",
    "suitable", "example", "examples", "types", "different", "need", "enlist",
    "various", "its", "how", "why", "role", "steps", "involved", "fundamental",
    "components", "characteristics", "following", "respect", "points", "give",
    "list", "typical", "between", "basis",
}

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def normalize_for_matching(text: str) -> str:
    """
    Normalizes exam command fluff & synonyms ONLY for similarity calculation.
    NEVER touches q.exact_text (which stays 100% verbatim).
    """
    s = text.lower().strip()

    # Map comparison synonyms to one canonical intent token
    s = re.sub(
        r"\b(?:compare\s+and\s+contrast|compare|distinguish\s+between|distinguish|"
        r"differentiate\s+between|differentiate|difference\s+between|differences\s+between|"
        r"contrast)\b",
        " _cmp_ ",
        s,
    )
    # Map architecture / diagram synonyms
    s = re.sub(
        r"\b(?:block\s+diagram|internal\s+architecture|architecture|constructional\s+diagram|"
        r"construction\s+and\s+working|construction)\b",
        " _arch_ ",
        s,
    )

    # Strip boilerplate command phrases that hide the real concept
    filler_patterns = [
        r"\bon\s+the\s+basis\s+of\s+features[,\s]+specifications\s+and\s+applications\b",
        r"\bwith\s+respect\s+to\s+the\s+following\s+points\b",
        r"\bwith\s+respect\s+to\b",
        r"\bwith\s+(?:a\s+)?(?:neat|suitable|labelled)?\s*diagram\b",
        r"\bwrite\s+(?:a\s+)?short\s+notes?\s+on\b",
        r"\b(?:draw\s+and\s+explain|list\s+and\s+explain|state\s+and\s+explain|identify\s+and\s+elaborate)\b",
        r"\b(?:explain|discuss|describe|define|draw|write|list|enlist|give|state|what\s+are|what\s+is)\b",
        r"\b(?:in\s+detail|in\s+brief|briefly|various|the\s+following|any\s+two\s+of\s+it)\b",
    ]
    for pat in filler_patterns:
        s = re.sub(pat, " ", s)

    s = re.sub(r"[^a-z0-9_\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or text.lower()


def chronological_paper_key(paper_name: str, question_number: str = "") -> tuple:
    low = paper_name.lower()
    year_m = re.search(r"\b(20\d{2})\b", low)
    year = int(year_m.group(1)) if year_m else 9999

    months_found = [
        MONTH_MAP[m[:3]]
        for m in re.findall(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", low)
    ]
    month = min(months_found) if months_found else 13

    q_m = re.match(r"Q\.(\d+)(?:\(([a-z]+)\))?", question_number, re.IGNORECASE)
    q_num = int(q_m.group(1)) if q_m else 99
    q_sub = (q_m.group(2) or "").lower() if q_m else ""

    return (year, month, low, q_num, q_sub)


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (arr / norms).astype(np.float32)


def _encode_via_hf_api(texts: List[str], token: str) -> np.ndarray | None:
    endpoints = [
        "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5/pipeline/feature-extraction",
        "https://api-inference.huggingface.co/pipeline/feature-extraction/BAAI/bge-small-en-v1.5",
    ]
    headers = {"Authorization": f"Bearer {token}"}
    for url in endpoints:
        try:
            resp = httpx.post(
                url,
                headers=headers,
                json={"inputs": texts, "options": {"wait_for_model": True}},
                timeout=30.0,
            )
            if resp.status_code == 200:
                data = np.asarray(resp.json(), dtype=np.float32)
                if data.ndim == 2 and data.shape[0] == len(texts):
                    return _l2_normalize(data)
        except Exception:
            continue
    return None


def _encode_tfidf_hybrid(texts: List[str]) -> np.ndarray:
    """High-precision TF-IDF word + char_wb hybrid for 512MB cloud tier."""
    word_vec = TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, stop_words="english"
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True
    )
    w_mat = word_vec.fit_transform(texts).toarray().astype(np.float32)
    c_mat = char_vec.fit_transform(texts).toarray().astype(np.float32)
    combined = np.hstack([w_mat * 1.4, c_mat * 0.6])
    return _l2_normalize(combined)


def get_embedding_model():
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None and SentenceTransformer is not None:
        _MODEL_INSTANCE = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _MODEL_INSTANCE


def encode_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)

    normed = [normalize_for_matching(t) for t in texts]

    hf_token = os.getenv("HF_TOKEN", "").strip()
    if hf_token:
        hf_embs = _encode_via_hf_api(normed, hf_token)
        if hf_embs is not None:
            return hf_embs

    if SentenceTransformer is not None:
        model = get_embedding_model()
        embeddings = model.encode(
            normed, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return embeddings.astype(np.float32)

    return _encode_tfidf_hybrid(normed)


def _tech_tokens(text: str) -> Set[str]:
    normed = normalize_for_matching(text)
    tokens = set(re.findall(r"[a-z0-9_]{2,}", normed))
    # Simple plural stem trim ('tools' -> 'tool', 'applications' -> 'application')
    stemmed = {t[:-1] if len(t) > 4 and t.endswith("s") else t for t in tokens}
    return stemmed - STOPWORDS


def compute_priority_tier(unique_papers: int, total_papers: int) -> PriorityTier:
    if total_papers >= 6:
        if unique_papers >= 5:
            return PriorityTier.HIGH
        if unique_papers >= 3:
            return PriorityTier.MEDIUM
        return PriorityTier.LOW
    if total_papers == 1:
        return PriorityTier.LOW

    high_cutoff = max(2, math.ceil(total_papers * 0.65))
    med_cutoff = max(2, math.ceil(total_papers * 0.4))
    if unique_papers >= high_cutoff:
        return PriorityTier.HIGH
    if unique_papers >= med_cutoff and total_papers >= 3:
        return PriorityTier.MEDIUM
    return PriorityTier.LOW


def cluster_questions(
    questions: List[ExtractedQuestion],
    embeddings: np.ndarray,
    threshold: float,
) -> List[Tuple[str, List[ExtractedQuestion], float]]:
    n = len(questions)
    if n == 0:
        return []
    if n == 1:
        return [(str(uuid.uuid4()), questions, 1.0)]

    if faiss is not None:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        sims, idxs = index.search(embeddings, n)
        sim_matrix = np.zeros((n, n), dtype=np.float32)
        row_indices = np.arange(n)[:, None]
        sim_matrix[row_indices, idxs] = sims
    else:
        sim_matrix = embeddings @ embeddings.T

    sim_matrix = np.clip((sim_matrix + sim_matrix.T) / 2.0, -1.0, 1.0)

    token_sets = [_tech_tokens(q.exact_text) for q in questions]
    dist_matrix = np.clip(1.0 - sim_matrix, 0.0, 2.0)

    for i in range(n):
        for j in range(i + 1, n):
            ui, uj = questions[i].unit_number, questions[j].unit_number
            if ui and uj and ui != uj:
                dist_matrix[i, j] = dist_matrix[j, i] = 2.0
                continue

            ti, tj = token_sets[i], token_sets[j]
            union = ti | tj
            jacc = (len(ti & tj) / len(union)) if union else 0.0
            hybrid_sim = 0.72 * float(sim_matrix[i, j]) + 0.28 * jacc
            dist = float(np.clip(1.0 - hybrid_sim, 0.0, 2.0))
            dist_matrix[i, j] = dist_matrix[j, i] = dist

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=1.0 - threshold,
    )
    labels = clustering.fit_predict(dist_matrix)

    grouped: Dict[int, List[int]] = {}
    for q_idx, lbl in enumerate(labels):
        grouped.setdefault(int(lbl), []).append(q_idx)

    results: List[Tuple[str, List[ExtractedQuestion], float]] = []
    for member_indices in grouped.values():
        member_qs = [questions[i] for i in member_indices]
        member_qs.sort(key=lambda q: chronological_paper_key(q.paper_name, q.question_number))

        if len(member_indices) > 1:
            sub = sim_matrix[np.ix_(member_indices, member_indices)]
            triu = sub[np.triu_indices(len(member_indices), k=1)]
            avg_sim = float(np.mean(triu))
        else:
            avg_sim = 1.0

        results.append((str(uuid.uuid4()), member_qs, round(avg_sim, 3)))

    results.sort(
        key=lambda item: (
            len({q.paper_id for q in item[1]}),
            len(item[1]),
            item[2],
        ),
        reverse=True,
    )
    return results

import math
import re
import uuid
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
from _lib.schemas import ExtractedQuestion, PriorityTier

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
    s = text.lower().strip()
    s = re.sub(
        r"\b(?:compare\s+and\s+contrast|compare|distinguish\s+between|distinguish|"
        r"differentiate\s+between|differentiate|difference\s+between|differences\s+between|contrast)\b",
        " _cmp_ ",
        s,
    )
    s = re.sub(
        r"\b(?:block\s+diagram|internal\s+architecture|architecture|constructional\s+diagram|"
        r"construction\s+and\s+working|construction)\b",
        " _arch_ ",
        s,
    )
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


def _tech_tokens(text: str) -> Set[str]:
    normed = normalize_for_matching(text)
    tokens = set(re.findall(r"[a-z0-9_]{2,}", normed))
    stemmed = {t[:-1] if len(t) > 4 and t.endswith("s") else t for t in tokens}
    return stemmed - STOPWORDS


def _doc_features(text: str) -> Counter:
    normed = normalize_for_matching(text)
    words = [w for w in normed.split() if w not in STOPWORDS and len(w) > 1]
    feats: Counter = Counter()
    for w in words:
        feats[f"w:{w}"] += 1.4
    for i in range(len(words) - 1):
        feats[f"b:{words[i]}_{words[i + 1]}"] += 1.4
    for w in words:
        padded = f" {w} "
        for n in (3, 4):
            if len(padded) >= n:
                for k in range(len(padded) - n + 1):
                    feats[f"c:{padded[k:k+n]}"] += 0.35
    return feats


def _build_tfidf_vectors(texts: List[str]) -> List[Dict[str, float]]:
    n = len(texts)
    raw_counts = [_doc_features(t) for t in texts]
    df: Counter = Counter()
    for c in raw_counts:
        for term in c.keys():
            df[term] += 1

    vectors: List[Dict[str, float]] = []
    for c in raw_counts:
        vec: Dict[str, float] = {}
        sq_sum = 0.0
        for term, tf in c.items():
            idf = math.log((1.0 + n) / (1.0 + df[term])) + 1.0
            w = (1.0 + math.log(tf)) * idf
            vec[term] = w
            sq_sum += w * w
        norm = math.sqrt(sq_sum) or 1.0
        for term in vec:
            vec[term] /= norm
        vectors.append(vec)
    return vectors


def _cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b[k] for k, v in a.items() if k in b)


def cluster_questions(
    questions: List[ExtractedQuestion],
    threshold: float,
) -> List[Tuple[str, List[ExtractedQuestion], float]]:
    """Pure-Python TF-IDF + Average-Linkage Clustering (zero numpy/sklearn)."""
    n = len(questions)
    if n == 0:
        return []
    if n == 1:
        return [(str(uuid.uuid4()), questions, 1.0)]

    vectors = _build_tfidf_vectors([q.exact_text for q in questions])
    token_sets = [_tech_tokens(q.exact_text) for q in questions]

    sim_matrix: List[List[float]] = [[1.0] * n for _ in range(n)]
    dist_matrix: List[List[float]] = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            ui, uj = questions[i].unit_number, questions[j].unit_number
            if ui and uj and ui != uj:
                sim_matrix[i][j] = sim_matrix[j][i] = 0.0
                dist_matrix[i][j] = dist_matrix[j][i] = 2.0
                continue

            cos_sim = _cosine_sparse(vectors[i], vectors[j])
            ti, tj = token_sets[i], token_sets[j]
            union = ti | tj
            jacc = (len(ti & tj) / len(union)) if union else 0.0
            hybrid_sim = max(0.0, min(1.0, 0.72 * cos_sim + 0.28 * jacc))
            sim_matrix[i][j] = sim_matrix[j][i] = cos_sim
            d = 1.0 - hybrid_sim
            dist_matrix[i][j] = dist_matrix[j][i] = d

    # Pure-Python Average-Linkage Agglomerative Clustering
    clusters: Dict[int, List[int]] = {i: [i] for i in range(n)}
    cluster_dists: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            cluster_dists[(i, j)] = dist_matrix[i][j]

    max_dist = 1.0 - threshold
    while len(clusters) > 1:
        best_pair = None
        best_d = max_dist + 1e-9
        keys = sorted(clusters.keys())
        for idx_a in range(len(keys)):
            ci = keys[idx_a]
            for idx_b in range(idx_a + 1, len(keys)):
                cj = keys[idx_b]
                d = cluster_dists[(ci, cj)]
                if d <= best_d:
                    best_d = d
                    best_pair = (ci, cj)
        if best_pair is None or best_d > max_dist:
            break

        ci, cj = best_pair
        si, sj = len(clusters[ci]), len(clusters[cj])
        clusters[ci].extend(clusters[cj])
        del clusters[cj]

        for ck in list(clusters.keys()):
            if ck == ci:
                continue
            p_ik = (ci, ck) if ci < ck else (ck, ci)
            p_jk = (cj, ck) if cj < ck else (ck, cj)
            d_ik = cluster_dists.pop(p_ik)
            d_jk = cluster_dists.pop(p_jk)
            new_d = (d_ik * si + d_jk * sj) / (si + sj)
            cluster_dists[(ci, ck) if ci < ck else (ck, ci)] = new_d

        # Remove any stale pairs referencing cj
        for pair in [p for p in cluster_dists if cj in p]:
            cluster_dists.pop(pair, None)

    results: List[Tuple[str, List[ExtractedQuestion], float]] = []
    for member_indices in clusters.values():
        member_qs = [questions[i] for i in member_indices]
        member_qs.sort(key=lambda q: chronological_paper_key(q.paper_name, q.question_number))
        if len(member_indices) > 1:
            pair_sims = [
                sim_matrix[member_indices[a]][member_indices[b]]
                for a in range(len(member_indices))
                for b in range(a + 1, len(member_indices))
            ]
            avg_sim = sum(pair_sims) / len(pair_sims)
        else:
            avg_sim = 1.0
        results.append((str(uuid.uuid4()), member_qs, round(avg_sim, 3)))

    results.sort(
        key=lambda item: (len({q.paper_id for q in item[1]}), len(item[1]), item[2]),
        reverse=True,
    )
    return results

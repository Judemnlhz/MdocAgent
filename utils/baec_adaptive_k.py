import re
from collections import defaultdict


RRF_C = 60
TASK_TYPE_PATTERNS = {
    "page_hint": [
        r"\bpage\s+\d+\b",
        r"\bpages\s+\d+",
    ],
    "counting": [
        r"\bhow many\b",
        r"\bnumber of\b",
        r"\bcount\b",
    ],
    "list": [
        r"\blist\b",
        r"\ball the\b",
        r"\bwhich two\b",
        r"\bwhat are the\b",
    ],
    "comparison": [
        r"\bincrease\b",
        r"\bdecrease\b",
        r"\bhigher\b",
        r"\blower\b",
        r"\bcompare\b",
        r"\bdifference\b",
        r"\btrend\b",
        r"\bmost\b",
        r"\bleast\b",
        r"\bgreater\b",
    ],
    "chart_value": [
        r"\bchart\b",
        r"\bfigure\b",
        r"\bgraph\b",
        r"\bimage\b",
        r"\bshown\b",
        r"\bdiagram\b",
        r"\bmap\b",
        r"\bcolor\b",
    ],
    "text_qa": [
        r"\baccording to the text\b",
        r"\bparagraph\b",
        r"\bsentence\b",
        r"\bauthor\b",
        r"\btitle\b",
        r"\bdefine\b",
        r"\bstand for\b",
    ],
}


def classify_baec_task_type(sample, question_key="question"):
    """Question-only task label for offline analysis; not used by selection."""
    question = str(sample.get(question_key, sample.get("question", ""))).lower()

    for task_type, patterns in TASK_TYPE_PATTERNS.items():
        if any(re.search(pattern, question) for pattern in patterns):
            return task_type
    return "general"


def parse_page_hints(question):
    q = str(question).lower()
    pages = set()

    for match in re.finditer(r"\bpage\s+(\d+)\b", q):
        pages.add(int(match.group(1)) - 1)

    for match in re.finditer(r"\bpages\s+(\d+)\s*(?:-|to)\s*(\d+)\b", q):
        start = int(match.group(1))
        end = int(match.group(2))
        if start > end:
            start, end = end, start
        for page in range(start, end + 1):
            pages.add(page - 1)

    return {page for page in pages if page >= 0}


def deduplicate_ranked_pages(pages):
    ranked_pages = []
    seen = set()
    for page in pages or []:
        try:
            page = int(page)
        except (TypeError, ValueError):
            continue
        if page < 0 or page in seen:
            continue
        ranked_pages.append(page)
        seen.add(page)
    return ranked_pages


def aggregate_page_scores(pages, scores):
    page2score = defaultdict(lambda: float("-inf"))

    for page, score in zip(pages or [], scores or []):
        try:
            page = int(page)
            score = float(score)
        except (TypeError, ValueError):
            continue
        if page < 0:
            continue
        page2score[page] = max(page2score[page], score)

    if pages and not scores:
        for page in pages:
            try:
                page = int(page)
            except (TypeError, ValueError):
                continue
            if page >= 0:
                page2score[page] = max(page2score[page], 0.0)

    return dict(page2score)


def build_rank_lookup(ranked_pages):
    return {page: rank for rank, page in enumerate(ranked_pages, start=1)}


def reciprocal_rank_fusion(rankers, rrf_c=RRF_C):
    candidate_pages = set()
    rank_lookups = {}
    for name, pages in rankers.items():
        lookup = build_rank_lookup(pages)
        rank_lookups[name] = lookup
        candidate_pages.update(lookup)

    rrf_scores = {}
    score_components = {}
    for page in candidate_pages:
        components = {}
        fused_score = 0.0
        for name, lookup in rank_lookups.items():
            rank = lookup.get(page)
            contribution = 0.0 if rank is None else 1.0 / (rrf_c + rank)
            fused_score += contribution
            components[name] = {
                "rank": rank,
                "rrf": round(contribution, 8),
            }
        rrf_scores[page] = fused_score
        score_components[str(page)] = components

    return rrf_scores, score_components


def rank_pages(rrf_scores, rank_prior):
    return sorted(
        rrf_scores,
        key=lambda page: (-rrf_scores[page], rank_prior.get(page, len(rank_prior)), page),
    )


def compute_score_gaps(ranked_pages, rrf_scores, k_max):
    search_limit = min(k_max + 1, len(ranked_pages))
    gaps = {}
    for idx in range(1, search_limit):
        left = ranked_pages[idx - 1]
        right = ranked_pages[idx]
        gaps[idx] = rrf_scores[left] - rrf_scores[right]
    return gaps


def select_k_by_largest_gap(ranked_pages, rrf_scores, k_max):
    if not ranked_pages or k_max <= 0:
        return 0, None, {}
    if len(ranked_pages) == 1 or k_max == 1:
        return 1, None, {}

    gaps = compute_score_gaps(ranked_pages, rrf_scores, k_max)
    if not gaps:
        return min(k_max, len(ranked_pages)), None, {}

    gap_index = max(gaps, key=lambda idx: (gaps[idx], -idx))
    used_k = max(1, min(k_max, gap_index))
    return used_k, gap_index, gaps


def merge_page_hints(selected_pages, ranked_pages, hinted_pages, k_max):
    if k_max <= 0:
        return []

    ranked_hints = [page for page in ranked_pages if page in hinted_pages]
    missing_hints = sorted(page for page in hinted_pages if page not in ranked_pages)
    ordered_hints = ranked_hints + missing_hints

    final_pages = []
    for page in ordered_hints + selected_pages:
        if page in final_pages:
            continue
        final_pages.append(page)
        if len(final_pages) >= k_max:
            break
    return final_pages


def page_hint_stats(hinted_pages, candidate_pool):
    if not hinted_pages:
        return False, []
    candidate_set = set(candidate_pool)
    outside_pool = sorted(page for page in hinted_pages if page not in candidate_set)
    return True, outside_pool


def merge_unique(*page_lists):
    merged = []
    for pages in page_lists:
        for page in pages:
            if page in merged:
                continue
            merged.append(page)
    return merged


def preferred_modality(task_type, sample):
    if task_type == "chart_value":
        return "image"
    if task_type == "text_qa":
        return "text"
    return None


def select_pages_baec(
    sample,
    k_max=4,
    rrf_c=RRF_C,
    question_key="question",
    text_key="text-top-10-question",
    image_key="image-top-10-question",
):
    question = sample.get(question_key, sample.get("question", ""))
    text_score_key = text_key + "_score"
    image_score_key = image_key + "_score"

    text_pages_dedup = deduplicate_ranked_pages(sample.get(text_key, []))
    image_pages_dedup = deduplicate_ranked_pages(sample.get(image_key, []))
    text_page_scores = aggregate_page_scores(sample.get(text_key, []), sample.get(text_score_key, []))
    image_page_scores = aggregate_page_scores(sample.get(image_key, []), sample.get(image_score_key, []))

    task_type = classify_baec_task_type(sample, question_key=question_key)
    hinted_pages = parse_page_hints(question)
    rrf_scores, score_components = reciprocal_rank_fusion(
        {
            "text": text_pages_dedup,
            "image": image_pages_dedup,
        },
        rrf_c=rrf_c,
    )

    for page in hinted_pages:
        rrf_scores.setdefault(page, 0.0)
        score_components.setdefault(
            str(page),
            {
                "text": {"rank": None, "rrf": 0.0},
                "image": {"rank": None, "rrf": 0.0},
            },
        )

    rank_prior = {}
    for idx, page in enumerate(text_pages_dedup):
        rank_prior[page] = min(rank_prior.get(page, idx), idx)
    for idx, page in enumerate(image_pages_dedup):
        rank_prior[page] = min(rank_prior.get(page, idx), idx)
    for page in hinted_pages:
        rank_prior[page] = min(rank_prior.get(page, len(rank_prior)), -1)

    rrf_candidate_pool = rank_pages(rrf_scores, rank_prior)[:10]
    if not rrf_candidate_pool:
        return _build_baec_result(
            task_type=task_type,
            selected_pages=[],
            candidate_pool=[],
            rrf_candidate_pool=[],
            rrf_scores={},
            score_gaps={},
            gap_index=None,
            score_components={},
            text_pages_dedup=text_pages_dedup,
            image_pages_dedup=image_pages_dedup,
            text_page_scores=text_page_scores,
            image_page_scores=image_page_scores,
            page_hints=[],
            k_max=k_max,
            rrf_c=rrf_c,
            sample=sample,
        )

    used_k, gap_index, score_gaps = select_k_by_largest_gap(rrf_candidate_pool, rrf_scores, k_max)
    selected_pages = rrf_candidate_pool[:used_k]
    selected_pages = merge_page_hints(selected_pages, rrf_candidate_pool, hinted_pages, k_max)
    candidate_pool = merge_unique(selected_pages, rrf_candidate_pool)

    return _build_baec_result(
        task_type=task_type,
        selected_pages=selected_pages,
        candidate_pool=candidate_pool,
        rrf_candidate_pool=rrf_candidate_pool,
        rrf_scores={str(page): round(rrf_scores[page], 8) for page in candidate_pool},
        score_gaps={str(idx): round(gap, 8) for idx, gap in score_gaps.items()},
        gap_index=gap_index,
        score_components=score_components,
        text_pages_dedup=text_pages_dedup,
        image_pages_dedup=image_pages_dedup,
        text_page_scores={str(page): round(score, 6) for page, score in text_page_scores.items()},
        image_page_scores={str(page): round(score, 6) for page, score in image_page_scores.items()},
        page_hints=sorted(hinted_pages),
        k_max=k_max,
        rrf_c=rrf_c,
        sample=sample,
    )


def _build_baec_result(
    task_type,
    selected_pages,
    candidate_pool,
    rrf_candidate_pool,
    rrf_scores,
    score_gaps,
    gap_index,
    score_components,
    text_pages_dedup,
    image_pages_dedup,
    text_page_scores,
    image_page_scores,
    page_hints,
    k_max,
    rrf_c,
    sample,
):
    page_hint_applied, page_hint_outside_candidate_pool = page_hint_stats(page_hints, rrf_candidate_pool)
    return {
        "baec_task_type": task_type,
        "baec_stage1": {
            "stage": "stage1_page_selection",
            "task_type_usage": "analysis_only",
            "action": "SELECT_PAGES",
            "page_hint_policy": "count_in_kmax",
            "selected_pages": selected_pages,
            "used_k": len(selected_pages),
            "k_max": k_max,
            "fusion_method": "RRF",
            "adaptive_k_method": "largest_gap",
            "page_hint_applied": page_hint_applied,
            "page_hint_outside_candidate_pool": page_hint_outside_candidate_pool,
        },
        "baec_controller": {
            "stage": "selection",
            "action": "SELECT_PAGES",
            "evidence_status": "unverified",
            "final_answer": None,
            "preferred_modality": preferred_modality(task_type, sample),
            "missing_info": None,
            "reason": (
                "Stage 1 selects pages using RRF-based multimodal rank fusion and "
                "Adaptive-k largest-gap cutoff. Evidence sufficiency, refusal, and "
                "iterative retrieval are not executed yet."
            ),
        },
        "baec_trace": {
            "round_id": 0,
            "fusion_method": "RRF",
            "adaptive_k_method": "largest_gap",
            "rrf_c": rrf_c,
            "k_max": k_max,
            "selected_pages": selected_pages,
            "used_k": len(selected_pages),
            "candidate_pool": candidate_pool,
            "rrf_candidate_pool": rrf_candidate_pool,
            "need_retrieve_more": False,
            "potential_need_retrieve_more": False,
            "text_pages_dedup": text_pages_dedup,
            "image_pages_dedup": image_pages_dedup,
            "rrf_scores": rrf_scores,
            "score_gaps": score_gaps,
            "gap_index": gap_index,
            "page_hints": page_hints,
            "page_hint_policy": "count_in_kmax",
            "page_hint_applied": page_hint_applied,
            "page_hint_outside_candidate_pool": page_hint_outside_candidate_pool,
            "score_components": score_components,
            "retriever_scores": {
                "text": text_page_scores,
                "image": image_page_scores,
            },
        },
        "baec_analysis": {
            "task_type": task_type,
            "task_type_usage": "analysis_only",
            "preferred_modality": preferred_modality(task_type, sample),
            "page_hint_applied": page_hint_applied,
            "page_hint_outside_candidate_pool": page_hint_outside_candidate_pool,
        },
    }

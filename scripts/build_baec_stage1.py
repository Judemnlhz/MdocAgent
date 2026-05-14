import json
import os
import sys
import ast
import re

import hydra
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.baec_adaptive_k import select_pages_baec


def parse_evidence_pages(value, one_based=True):
    if value is None:
        return set()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return set()
        try:
            value = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            value = [int(match) for match in re.findall(r"\d+", text)]
    if isinstance(value, int):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        return set()

    pages = set()
    for page in value:
        try:
            page = int(page)
        except (TypeError, ValueError):
            continue
        if one_based and page > 0:
            page -= 1
        if page >= 0:
            pages.add(page)
    return pages


def min_rank(pages, ranked_pages):
    ranks = {page: idx for idx, page in enumerate(ranked_pages, start=1)}
    found = [ranks[page] for page in pages if page in ranks]
    return min(found) if found else None


def update_evidence_stats(stats, sample, baec_result, one_based=True):
    evidence_pages = parse_evidence_pages(sample.get("evidence_pages"), one_based=one_based)
    if not evidence_pages:
        return

    trace = baec_result["baec_trace"]
    selected_pages = set(trace.get("selected_pages", []))
    rrf_candidate_pool = trace.get("rrf_candidate_pool", trace.get("candidate_pool", []))
    fixed_top1_pages = set(rrf_candidate_pool[:1])
    fixed_top4_pages = set(rrf_candidate_pool[:4])
    hits = selected_pages & evidence_pages
    fixed_top1_hits = fixed_top1_pages & evidence_pages
    fixed_top4_hits = fixed_top4_pages & evidence_pages
    stats["count"] += 1
    stats["recall_sum"] += len(hits) / len(evidence_pages)
    stats["precision_sum"] += len(hits) / len(selected_pages) if selected_pages else 0.0
    stats["fixed_top1_recall_sum"] += len(fixed_top1_hits) / len(evidence_pages)
    stats["fixed_top1_precision_sum"] += len(fixed_top1_hits) / len(fixed_top1_pages) if fixed_top1_pages else 0.0
    stats["fixed_top4_recall_sum"] += len(fixed_top4_hits) / len(evidence_pages)
    stats["fixed_top4_precision_sum"] += len(fixed_top4_hits) / len(fixed_top4_pages) if fixed_top4_pages else 0.0

    text_rank = min_rank(evidence_pages, trace.get("text_pages_dedup", []))
    image_rank = min_rank(evidence_pages, trace.get("image_pages_dedup", []))
    retriever_ranks = [rank for rank in [text_rank, image_rank] if rank is not None]
    if retriever_ranks:
        stats["before_rank_sum"] += min(retriever_ranks)
        stats["before_rank_count"] += 1
    rrf_rank = min_rank(evidence_pages, rrf_candidate_pool)
    if rrf_rank is not None:
        stats["rrf_rank_sum"] += rrf_rank
        stats["rrf_rank_count"] += 1


def average(total, count):
    return round(total / count, 6) if count else None


def resolve_baec_output_path(cfg):
    input_path = cfg.dataset.sample_with_retrieval_path
    output_path = cfg.dataset.get("baec_output_path")
    if output_path:
        return output_path
    if input_path.endswith(".json"):
        return input_path[:-5] + ".baec_stage1.json"
    return input_path + ".baec_stage1.json"


@hydra.main(config_path="../config", config_name="base", version_base="1.2")
def main(cfg):
    input_path = cfg.dataset.sample_with_retrieval_path
    output_path = resolve_baec_output_path(cfg)
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError(
            "dataset.baec_output_path must not be the same as "
            "dataset.sample_with_retrieval_path; Stage 1 must not overwrite retrieval JSON"
        )
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as file:
        samples = json.load(file)

    k_max = cfg.dataset.get("baec_k_max", cfg.dataset.top_k)
    rrf_c = cfg.dataset.get("baec_rrf_c", 60)
    k_stats = {}
    task_stats = {}
    gap_stats = {}
    evidence_stats = {
        "count": 0,
        "recall_sum": 0.0,
        "precision_sum": 0.0,
        "fixed_top1_recall_sum": 0.0,
        "fixed_top1_precision_sum": 0.0,
        "fixed_top4_recall_sum": 0.0,
        "fixed_top4_precision_sum": 0.0,
        "before_rank_sum": 0.0,
        "before_rank_count": 0,
        "rrf_rank_sum": 0.0,
        "rrf_rank_count": 0,
    }
    page_hint_applied_count = 0
    page_hint_outside_candidate_pool_count = 0
    evidence_pages_one_based = cfg.dataset.get("baec_evidence_pages_one_based", True)

    for sample in tqdm(samples):
        baec_result = select_pages_baec(
            sample,
            k_max=k_max,
            rrf_c=rrf_c,
            question_key=cfg.dataset.question_key,
            text_key=cfg.dataset.r_text_key,
            image_key=cfg.dataset.r_image_key,
        )
        sample.update(baec_result)

        used_k = baec_result["baec_trace"]["used_k"]
        task_type = baec_result["baec_analysis"]["task_type"]
        gap_index = baec_result["baec_trace"]["gap_index"]
        page_hint_applied_count += int(baec_result["baec_stage1"]["page_hint_applied"])
        page_hint_outside_candidate_pool_count += int(
            bool(baec_result["baec_stage1"]["page_hint_outside_candidate_pool"])
        )
        k_stats[used_k] = k_stats.get(used_k, 0) + 1
        task_stats[task_type] = task_stats.get(task_type, 0) + 1
        gap_stats[gap_index] = gap_stats.get(gap_index, 0) + 1
        update_evidence_stats(
            evidence_stats,
            sample,
            baec_result,
            one_based=evidence_pages_one_based,
        )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(samples, file, indent=4, ensure_ascii=False)

    print("Loaded retrieval JSON from:", input_path)
    print("Saved BAEC Stage 1 fields to:", output_path)
    print("fusion method: RRF")
    print("adaptive-k method: largest_gap")
    print("task_type_usage: analysis_only")
    print("action: SELECT_PAGES")
    print("page_hint_policy: count_in_kmax")
    print("rrf_c:", rrf_c)
    print("k_max:", k_max)
    print("used_k distribution:", dict(sorted(k_stats.items())))
    print("avg_used_k:", round(sum(k * v for k, v in k_stats.items()) / len(samples), 6))
    print("task_type distribution:", dict(sorted(task_stats.items())))
    print("gap_index distribution:", dict(sorted(gap_stats.items(), key=lambda item: str(item[0]))))
    print("page_hint_applied count:", page_hint_applied_count)
    print("page_hint_outside_candidate_pool count:", page_hint_outside_candidate_pool_count)
    if evidence_stats["count"]:
        print("evidence samples:", evidence_stats["count"])
        print("evidence recall@selected:", average(evidence_stats["recall_sum"], evidence_stats["count"]))
        print("evidence precision@selected:", average(evidence_stats["precision_sum"], evidence_stats["count"]))
        print("fixed top1 recall:", average(evidence_stats["fixed_top1_recall_sum"], evidence_stats["count"]))
        print("fixed top1 precision:", average(evidence_stats["fixed_top1_precision_sum"], evidence_stats["count"]))
        print(
            "fixed top4 retrieval-only recall:",
            average(evidence_stats["fixed_top4_recall_sum"], evidence_stats["count"]),
        )
        print(
            "fixed top4 retrieval-only precision:",
            average(evidence_stats["fixed_top4_precision_sum"], evidence_stats["count"]),
        )
        print(
            "gold evidence best rank before RRF:",
            average(evidence_stats["before_rank_sum"], evidence_stats["before_rank_count"]),
        )
        print(
            "gold evidence best rank after RRF:",
            average(evidence_stats["rrf_rank_sum"], evidence_stats["rrf_rank_count"]),
        )


if __name__ == "__main__":
    main()

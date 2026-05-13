import json
import re
from typing import Any, Dict


REFUSAL_MARKERS = (
    "not answerable",
    "cannot be determined",
    "can not be determined",
    "cannot answer",
    "can't answer",
    "insufficient information",
    "not enough information",
    "no sufficient evidence",
    "unknown",
)


def route_question_type(question: str) -> str:
    question_lower = question.lower()
    if any(token in question_lower for token in ["how many", "number of", "count"]):
        return "counting"
    if any(
        token in question_lower
        for token in [
            "sum",
            "difference",
            "drop",
            "increase",
            "decrease",
            "greater",
            "higher",
            "lower",
            "calculate",
        ]
    ):
        return "calculation"
    if any(token in question_lower for token in ["list", "what are", "which two", "examples"]):
        return "list"
    if any(token in question_lower for token in ["chart", "figure", "map", "color", "shape", "picture"]):
        return "visual"
    if any(token in question_lower for token in ["table", "percentage", "%", "rate", "value"]):
        return "numeric"
    if any(token in question_lower for token in ["yes or no", "answer 'yes' or 'no'", "whether"]):
        return "yes_no"
    return "standard"


def is_refusal_answer(answer: Any) -> bool:
    if answer is None:
        return True
    normalized = str(answer).strip().lower()
    if not normalized:
        return True
    return any(marker in normalized for marker in REFUSAL_MARKERS)


def build_verification_prompt(
    question: str,
    question_type: str,
    candidate_answer: Any,
    agent_evidence: str,
    refuse_answer: str,
) -> str:
    return f"""
You are an evidence sufficiency controller for multimodal document question answering.
Your task is to verify whether the candidate answer is directly supported by the agent evidence.

Question type: {question_type}
Question:
{question}

Candidate answer:
{candidate_answer}

Agent evidence:
{agent_evidence}

Choose exactly one action:
1. ANSWER: the evidence directly supports the candidate answer, or directly supports a corrected final answer.
2. RETRIEVE_MORE: the evidence is partially relevant but not sufficient to answer.
3. REFUSE: the evidence does not support an answer, is contradictory, or the question is not answerable from the given evidence.

Rules:
- Do not guess.
- If the answer is not explicitly supported by the evidence, choose RETRIEVE_MORE or REFUSE.
- If the candidate answer uses uncertain language such as "probably", "may", "at least", or "not provided", choose REFUSE unless the final answer is exactly a refusal.
- For counting questions, counted items or an explicit count must be present in the evidence.
- For calculation questions, all required values and the operation must be present in the evidence.
- For list questions, every item in the final answer must be supported by the evidence.
- For yes/no questions, the evidence must directly support the yes/no decision.
- If you choose RETRIEVE_MORE or REFUSE, set final_answer to "{refuse_answer}".

Return only valid JSON in this format:
{{"action": "ANSWER/RETRIEVE_MORE/REFUSE", "final_answer": "...", "reason": "..."}}
""".strip()


def parse_verification_response(response: Any) -> Dict[str, Any]:
    text = "" if response is None else str(response)
    data = _extract_json_object(text)
    action = str(data.get("action", "")).strip().upper()
    if action not in {"ANSWER", "RETRIEVE_MORE", "REFUSE"}:
        action = "INVALID"
    return {
        "action": action,
        "final_answer": data.get("final_answer", ""),
        "reason": data.get("reason", ""),
        "raw_response": text,
    }


def apply_verification_decision(
    candidate_answer: Any,
    decision: Dict[str, Any],
    refuse_answer: str,
) -> Any:
    if is_refusal_answer(candidate_answer):
        return refuse_answer

    action = decision.get("action", "INVALID")
    final_answer = decision.get("final_answer", "")

    if action == "ANSWER":
        if is_refusal_answer(final_answer):
            return refuse_answer
        return final_answer or candidate_answer

    if action in {"RETRIEVE_MORE", "REFUSE"}:
        return refuse_answer

    return candidate_answer


def _extract_json_object(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    json_text = text[start : end + 1]
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        pass

    compact_json_text = re.sub(r",\s*}", "}", json_text)
    try:
        return json.loads(compact_json_text)
    except json.JSONDecodeError:
        return {}

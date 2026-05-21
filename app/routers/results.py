# app/routers/results.py

from fastapi import APIRouter, Query
from typing import Any, Dict, List, Optional
import re
import json

from pyrit.memory import CentralMemory

router = APIRouter()


# ---------------------------
# Utilities
# ---------------------------
def _to_str(val: Any, default: str = "") -> str:
    return val if isinstance(val, str) else default


def _dedup_by_id(rows) -> List[Any]:
    """De-duplicate by piece id (authoritative)."""
    dedup = {}
    for p in rows:
        dedup[p.id] = p
    return list(dedup.values())


def _dedup_by_value_hash(rows: List[Any]) -> List[Any]:
    """
    Collapse exact text repeats using original_value_sha256 per (conversation_id, role).
    Keep first occurrence (stable order).
    """
    seen = set()
    out = []
    for p in rows:
        h = getattr(p, "original_value_sha256", None)
        key = (getattr(p, "conversation_id", None), getattr(p, "role", None), h)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _collapse_consecutive_duplicates(rows: List[Any]) -> List[Any]:
    """
    Remove runs of consecutive pieces that have the same role and exact same text.
    Useful when the same content is written multiple times in sequence.
    """
    out: List[Any] = []
    prev_key = None
    for p in rows:
        key = (getattr(p, "conversation_id", None),
               getattr(p, "role", None),
               getattr(p, "original_value_sha256", None))
        if key == prev_key:
            continue
        out.append(p)
        prev_key = key
    return out


def _gather_all_pieces_for_run(mem, run_id: str) -> List[Any]:
    """
    1) Get labeled pieces for run_id
    2) Collect conversation_ids
    3) Pull all pieces in those conversations
    4) De-duplicate by id
    """
    labeled = mem.get_message_pieces(labels={"run_id": run_id})
    conv_ids = {p.conversation_id for p in labeled if p.conversation_id}
    all_rows = list(labeled)
    for cid in conv_ids:
        all_rows.extend(mem.get_message_pieces(conversation_id=cid))
    return _dedup_by_id(all_rows)


def _stable_sort(rows: List[Any]) -> List[Any]:
    rows.sort(
        key=lambda r: (
            getattr(r, "conversation_id", "") or "",
            getattr(r, "sequence", 0) or 0,
            _to_str(getattr(r, "timestamp", ""), "")
        )
    )
    return rows


def _piece_model_type(piece: Any) -> str:
    pti = getattr(piece, "prompt_target_identifier", {}) or {}
    return pti.get("__type__", "") if isinstance(pti, dict) else ""


def _first_score(piece: Any) -> Optional[Dict[str, Any]]:
    scores = getattr(piece, "scores", None)
    if not scores:
        return None
    s = scores[0]
    return {
        "id": getattr(s, "id", None),
        "score_value": getattr(s, "score_value", None),
        "score_type": getattr(s, "score_type", None),
        "score_rationale": getattr(s, "score_rationale", None),
        "objective": getattr(s, "objective", None),
    }


# ---------------------------
# JSON extraction for clean view
# ---------------------------
def _is_valid_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except Exception:
        return False


def _extract_balanced_json_object(text: str) -> Optional[str]:
    """Return first balanced {...} JSON object or None."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    i = start
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1].strip()
                if _is_valid_json(candidate):
                    return candidate
                # try another object in the remainder
                remainder = text[i + 1 :]
                nxt = _extract_balanced_json_object(remainder)
                return nxt
        i += 1
    return None


def _to_strict_json_or_none(text: str) -> Optional[str]:
    """Best-effort JSON object normalization from text; None if nothing valid."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.IGNORECASE | re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
        if _is_valid_json(candidate):
            return candidate
    candidate = _extract_balanced_json_object(text)
    if candidate:
        return candidate
    return None


# =================================================================
# 1) RAW VIEW (DEFAULT DEDUPE + COLLAPSE): GET /results/{run_id}
# =================================================================
@router.get("/{run_id}")
async def get_results(
    run_id: str,
    dedupe: bool = Query(default=True, description="If true, collapse repeats by value hash per conversation+role."),
    collapse_runs: bool = Query(default=True, description="If true, remove consecutive duplicates.")
):
    mem = CentralMemory.get_memory_instance()
    rows = _gather_all_pieces_for_run(mem, run_id)
    rows = _stable_sort(rows)

    if dedupe:
        rows = _dedup_by_value_hash(rows)
    if collapse_runs:
        rows = _collapse_consecutive_duplicates(rows)

    return rows


# =================================================================
# 2) OBJECTIVE VIEW: GET /results/{run_id}/objective
#    - ONLY assistant turns that have a score (good proxy for “objective response”)
#    - Dedupe + collapse
# =================================================================
@router.get("/{run_id}/objective")
async def get_results_objective(
    run_id: str,
    dedupe: bool = Query(default=True),
    collapse_runs: bool = Query(default=True),
):
    mem = CentralMemory.get_memory_instance()
    rows = _gather_all_pieces_for_run(mem, run_id)
    rows = _stable_sort(rows)

    # keep assistant turns that have a score attached
    filtered = []
    for p in rows:
        if getattr(p, "role", "") != "assistant":
            continue
        if _first_score(p) is None:
            continue
        filtered.append(p)

    if dedupe:
        filtered = _dedup_by_value_hash(filtered)
    if collapse_runs:
        filtered = _collapse_consecutive_duplicates(filtered)

    # return a compact projection
    out = []
    for p in filtered:
        out.append({
            "id": getattr(p, "id", None),
            "conversation_id": getattr(p, "conversation_id", None),
            "text": getattr(p, "original_value", ""),
            "score": _first_score(p),
        })
    return out


# =================================================================
# 3) CLEAN JSON VIEW: GET /results/{run_id}/clean
#    - assistant turns where we can produce JSON (prefer evaluator)
#    - derive JSON from score if missing
# =================================================================
@router.get("/{run_id}/clean")
async def get_results_clean(run_id: str):
    mem = CentralMemory.get_memory_instance()
    rows = _gather_all_pieces_for_run(mem, run_id)
    rows = _stable_sort(rows)

    clean: List[Dict[str, Any]] = []
    seen_ids = set()

    for p in rows:
        pid = getattr(p, "id", None)
        if not pid or pid in seen_ids:
            continue

        role = getattr(p, "role", "")
        if role != "assistant":
            continue

        text = getattr(p, "original_value", "") or ""
        model_type = _piece_model_type(p)

        json_text: Optional[str] = None
        if model_type == "JsonEnforcedChatTarget":
            json_text = text if _is_valid_json(text) else _to_strict_json_or_none(text)
        else:
            json_text = _to_strict_json_or_none(text)

        if not json_text:
            score = _first_score(p)
            if score and score.get("score_value") is not None:
                json_text = json.dumps({
                    "score_value": str(score["score_value"]).lower(),
                    "rationale": score.get("score_rationale", "")
                })

        if not json_text:
            continue

        clean.append({
            "piece_id": pid,
            "conversation_id": getattr(p, "conversation_id", None),
            "model_type": model_type,
            "json": json.loads(json_text),
            "score": _first_score(p),
        })
        seen_ids.add(pid)

    return clean


# =================================================================
# 4) SUMMARY VIEW: GET /results/{run_id}/summary
#    - Groups by conversation_id
#    - Shows user -> assistant pairs; attaches score if present
# =================================================================
@router.get("/{run_id}/summary")
async def get_results_summary(run_id: str):
    mem = CentralMemory.get_memory_instance()
    rows = _gather_all_pieces_for_run(mem, run_id)
    rows = _stable_sort(rows)

    by_conv: Dict[str, Dict[str, List[Any]]] = {}
    for p in rows:
        cid = getattr(p, "conversation_id", None) or "unknown"
        by_conv.setdefault(cid, {"user": [], "assistant": [], "other": []})
        role = getattr(p, "role", "")
        if role in ("user", "assistant"):
            by_conv[cid][role].append(p)
        else:
            by_conv[cid]["other"].append(p)

    summary: List[Dict[str, Any]] = []
    for cid, parts in by_conv.items():
        users = _collapse_consecutive_duplicates(_dedup_by_value_hash(parts["user"]))
        assistants = _collapse_consecutive_duplicates(_dedup_by_value_hash(parts["assistant"]))

        max_len = max(len(users), len(assistants))
        for i in range(max_len):
            u = users[i] if i < len(users) else None
            a = assistants[i] if i < len(assistants) else None

            summary.append({
                "conversation_id": cid,
                "user_text": getattr(u, "original_value", None) if u else None,
                "assistant_text": getattr(a, "original_value", None) if a else None,
                "assistant_model_type": _piece_model_type(a) if a else "",
                "score": _first_score(a) if a else None,
            })

    return summary
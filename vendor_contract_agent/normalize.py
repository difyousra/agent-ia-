from __future__ import annotations

import copy
import json
import re
from typing import Any

from .schema import (
    ACTION_ITEM,
    DATE_ITEM,
    EMPTY_SCHEMA,
    PENALTY_ITEM,
    RISK_ITEM,
    SOURCE_ITEM,
)

LIST_ITEM_TEMPLATES = {
    "penalties": PENALTY_ITEM,
    "important_dates": DATE_ITEM,
    "actions": ACTION_ITEM,
    "risks": RISK_ITEM,
    "sources": SOURCE_ITEM,
}


def extract_json_payload(raw: str) -> dict[str, Any]:
    candidate = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("La réponse du modèle ne contient pas de JSON.")
        candidate = candidate[start : end + 1]
    data = json.loads(candidate)
    if not isinstance(data, dict):
        raise ValueError("Le JSON extrait n'est pas un objet.")
    return data


def normalize_result(data: dict[str, Any]) -> dict[str, Any]:
    return _merge(EMPTY_SCHEMA, data)


def _merge(template: Any, value: Any) -> Any:
    if isinstance(template, dict):
        source = value if isinstance(value, dict) else {}
        merged = {}
        for key, default in template.items():
            merged[key] = _merge(default, source.get(key))
        return merged

    if isinstance(template, list):
        if not isinstance(value, list):
            return []
        cleaned = [item for item in value if not _is_empty_placeholder(item)]
        return cleaned

    if value is None:
        return None
    return value


def normalize_list_items(result: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(result)
    for key, item_template in LIST_ITEM_TEMPLATES.items():
        items = output.get(key)
        if not isinstance(items, list):
            output[key] = []
            continue
        normalized_items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if _is_empty_placeholder(item):
                continue
            normalized_items.append(_merge(item_template, item))
        output[key] = normalized_items
    return output


def _is_empty_placeholder(item: Any) -> bool:
    if item is None:
        return True
    if isinstance(item, dict):
        return all(_is_empty_placeholder(v) for v in item.values())
    if isinstance(item, list):
        return len(item) == 0 or all(_is_empty_placeholder(v) for v in item)
    return False

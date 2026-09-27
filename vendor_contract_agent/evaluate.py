from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .atticus_gt import (
    EXTRACTIVE_CATEGORIES,
    YES_NO_CATEGORIES,
    find_clause_csv,
    load_atticus_labels,
    normalize_date,
    normalize_text,
    party_names_from_atticus,
)
from .dataset import select_test_contracts


def _filled(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_filled(item) for item in value)
    if isinstance(value, dict):
        return any(_filled(item) for item in value.values())
    return True


def _all_text(agent: dict[str, Any]) -> str:
    return normalize_text(json.dumps(agent, ensure_ascii=False))


def agent_predictions(agent: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Projette la sortie agent sur les catégories Atticus évaluables."""
    contract = agent.get("contract") or {}
    parties = agent.get("parties") or {}
    vendor = (parties.get("vendor") or {}).get("name")
    customer = (parties.get("customer") or {}).get("name")
    renewal = agent.get("renewal") or {}
    termination = agent.get("termination") or {}
    governing = agent.get("governing_law") or {}
    audit = agent.get("audit") or {}
    liability = agent.get("liability") or {}
    ip = agent.get("intellectual_property") or {}
    insurance = agent.get("insurance") or {}
    confidentiality = agent.get("confidentiality") or {}
    financial = agent.get("financial") or {}
    blob = _all_text(agent)

    party_list = [p for p in [vendor, customer] if p]
    party_answer = "; ".join(party_list) if party_list else None

    agreement_date = contract.get("effective_date")
    effective_date = contract.get("start_date") or contract.get("effective_date")
    expiration_date = contract.get("end_date")

    notice_days = renewal.get("notice_period_days")
    notice_answer = f"{notice_days} days" if notice_days is not None else None

    law_parts = [governing.get("law"), governing.get("country")]
    law_answer = ", ".join(p for p in law_parts if p) or None

    predictions = {
        "Document Name": {
            "answer": contract.get("contract_title") or contract.get("contract_type"),
            "present": _filled(contract.get("contract_title") or contract.get("contract_type")),
        },
        "Parties": {
            "answer": party_answer,
            "present": bool(party_list),
            "party_names": party_list,
        },
        "Agreement Date": {
            "answer": agreement_date,
            "present": _filled(agreement_date),
        },
        "Effective Date": {
            "answer": effective_date,
            "present": _filled(effective_date),
        },
        "Expiration Date": {
            "answer": expiration_date,
            "present": _filled(expiration_date),
        },
        "Renewal Term": {
            "answer": renewal.get("renewal_period"),
            "present": _filled(renewal.get("renewal_period")),
        },
        "Notice Period To Terminate Renewal": {
            "answer": notice_answer,
            "present": notice_days is not None,
        },
        "Governing Law": {
            "answer": law_answer,
            "present": _filled(law_answer),
        },
        "Termination for Convenience": {
            "answer": "Yes" if _filled(termination.get("for_convenience")) else "No",
            "present": _filled(termination.get("for_convenience")),
        },
        "Audit Rights": {
            "answer": "Yes" if audit.get("allowed") is True else "No",
            "present": audit.get("allowed") is True,
        },
        "Uncapped Liability": {
            "answer": "Yes" if _filled(liability.get("uncapped_liabilities")) else "No",
            "present": _filled(liability.get("uncapped_liabilities")),
        },
        "Cap On Liability": {
            "answer": "Yes" if _filled(liability.get("liability_cap")) else "No",
            "present": _filled(liability.get("liability_cap")),
        },
        "Insurance": {
            "answer": "Yes" if insurance.get("required") is True or _filled(insurance.get("types")) else "No",
            "present": insurance.get("required") is True or _filled(insurance.get("types")),
        },
        "License Grant": {
            "answer": "Yes" if ("license" in blob or _filled(ip.get("license"))) else "No",
            "present": "license" in blob or _filled(ip.get("license")),
        },
        "Post-Termination Services": {
            "answer": "Yes"
            if any(
                "transition" in normalize_text(str(item))
                or "post termination" in normalize_text(str(item))
                or "after termination" in normalize_text(str(item))
                for item in (termination.get("consequences") or [])
            )
            else "No",
            "present": any(
                "transition" in normalize_text(str(item))
                or "post termination" in normalize_text(str(item))
                or "after termination" in normalize_text(str(item))
                for item in (termination.get("consequences") or [])
            ),
        },
        "Anti-assignment": {
            "answer": "Yes" if "assign" in blob or "assignment" in blob else "No",
            "present": "assign" in blob or "assignment" in blob,
        },
        "Revenue/Profit Sharing": {
            "answer": "Yes"
            if any(
                token in blob
                for token in ["profit sharing", "revenue share", "royalt", "profit share"]
            )
            else "No",
            "present": any(
                token in blob
                for token in ["profit sharing", "revenue share", "royalt", "profit share"]
            ),
        },
        "Liquidated Damages": {
            "answer": "Yes" if _filled(agent.get("penalties")) else "No",
            "present": _filled(agent.get("penalties")),
        },
    }

    # Confidentiality is not an Atticus top-level category name in all files,
    # but keep extract for reporting when present in agent.
    if confidentiality.get("applicable") is not None:
        predictions["Confidentiality"] = {
            "answer": "Yes" if confidentiality.get("applicable") else "No",
            "present": bool(confidentiality.get("applicable")),
        }
    return predictions


def _dates_match(gt: str | None, pred: str | None) -> bool:
    left = normalize_date(gt)
    right = normalize_date(pred)
    if not left or not right:
        return False
    return left == right


def _names_match(gt_names: list[str], pred_names: list[str]) -> bool:
    if not gt_names or not pred_names:
        return False
    pred_norm = [normalize_text(name) for name in pred_names]
    hits = 0
    for gt in gt_names:
        gt_n = normalize_text(gt)
        if any(gt_n in pred or pred in gt_n for pred in pred_norm if pred and gt_n):
            hits += 1
    return hits >= min(2, len(gt_names)) if len(gt_names) >= 2 else hits >= 1


def _text_match(gt: str | None, pred: str | None) -> bool:
    left = normalize_text(gt)
    right = normalize_text(pred)
    if not left or not right:
        return False
    if left == right:
        return True
    if left in right or right in left:
        return True
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens) / len(left_tokens)
    return overlap >= 0.6


def _notice_match(gt: str | None, pred: str | None) -> bool:
    if not gt or not pred:
        return False
    gt_days = re.search(r"(\d+)", gt)
    pred_days = re.search(r"(\d+)", pred)
    if gt_days and pred_days:
        return gt_days.group(1) == pred_days.group(1)
    return _text_match(gt, pred)


def _category_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _lookup(mapping: dict[str, Any], category: str) -> Any | None:
    if category in mapping:
        return mapping[category]
    target = _category_key(category)
    for key, value in mapping.items():
        if _category_key(key) == target:
            return value
    return None


def _renewal_match(gt: str | None, pred: str | None) -> bool:
    if _text_match(gt, pred):
        return True
    if not gt or not pred:
        return False
    left = normalize_text(gt)
    right = normalize_text(pred)
    aliases = {
        "0 5 year": {"0 5 year", "0 5 years", "half year", "half a year", "6 months", "six months"},
        "half a year": {"0 5 year", "0 5 years", "half year", "half a year", "6 months", "six months"},
    }
    for group in aliases.values():
        if left in group and right in group:
            return True
    return False


def compare_category(category: str, gt: dict | None, pred: dict | None) -> dict[str, Any]:
    gt = gt or {"answer": None, "present": False, "answers": [], "clauses": []}
    pred = pred or {"answer": None, "present": False}

    yes_no_keys = {_category_key(name) for name in YES_NO_CATEGORIES} | {
        _category_key(name)
        for name in {
            "Termination for Convenience",
            "Audit Rights",
            "Uncapped Liability",
            "Cap On Liability",
            "Insurance",
            "License Grant",
            "Post-Termination Services",
            "Post-termination Services",
            "Anti-assignment",
            "Revenue/Profit Sharing",
            "Liquidated Damages",
        }
    }

    if _category_key(category) in yes_no_keys:
        gt_present = bool(gt.get("present"))
        pred_present = bool(pred.get("present"))
        return {
            "category": category,
            "type": "presence",
            "ground_truth": "Yes" if gt_present else "No",
            "agent": "Yes" if pred_present else "No",
            "match": gt_present == pred_present,
            "gt_answer": gt.get("answer"),
            "agent_answer": pred.get("answer"),
        }

    gt_answer = gt.get("answer")
    agent_answer = pred.get("answer")
    matched = False
    if _category_key(category) == _category_key("Parties"):
        matched = _names_match(
            party_names_from_atticus(gt_answer),
            pred.get("party_names") or [],
        )
    elif _category_key(category) in {
        _category_key("Agreement Date"),
        _category_key("Effective Date"),
        _category_key("Expiration Date"),
    }:
        matched = _dates_match(gt_answer, agent_answer)
    elif _category_key(category) == _category_key("Notice Period To Terminate Renewal"):
        matched = _notice_match(gt_answer, agent_answer)
    elif _category_key(category) == _category_key("Renewal Term"):
        matched = _renewal_match(gt_answer, agent_answer)
    else:
        matched = _text_match(gt_answer, agent_answer)

    return {
        "category": category,
        "type": "extractive",
        "ground_truth": gt_answer,
        "agent": agent_answer,
        "match": matched,
        "gt_answer": gt_answer,
        "agent_answer": agent_answer,
    }


def evaluate_contract(contract_pdf: Path, agent_result: dict[str, Any]) -> dict[str, Any]:
    csv_path = find_clause_csv(contract_pdf)
    ground_truth = load_atticus_labels(csv_path)
    predictions = agent_predictions(agent_result)

    # Compare only categories annotated in the dataset for this contract.
    comparable = [
        compare_category(
            category,
            ground_truth[category],
            _lookup(predictions, category),
        )
        for category in sorted(ground_truth.keys())
    ]

    matches = sum(1 for item in comparable if item["match"])
    total = len(comparable)
    extractive = [item for item in comparable if item["type"] == "extractive"]
    presence = [item for item in comparable if item["type"] == "presence"]

    return {
        "file": contract_pdf.name,
        "annotation_csv": csv_path.name,
        "ground_truth_labels": {
            key: {
                "answer": value.get("answer"),
                "present": value.get("present"),
            }
            for key, value in ground_truth.items()
        },
        "comparisons": comparable,
        "metrics": {
            "total": total,
            "matches": matches,
            "accuracy": round(matches / total, 4) if total else None,
            "extractive_total": len(extractive),
            "extractive_matches": sum(1 for item in extractive if item["match"]),
            "presence_total": len(presence),
            "presence_matches": sum(1 for item in presence if item["match"]),
        },
    }


def evaluate_selected(
    agent_outputs: list[tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    per_contract = []
    for pdf_path, agent_result in agent_outputs:
        per_contract.append(evaluate_contract(pdf_path, agent_result))

    totals = sum(item["metrics"]["total"] for item in per_contract)
    matches = sum(item["metrics"]["matches"] for item in per_contract)
    return {
        "contracts_evaluated": len(per_contract),
        "overall": {
            "total": totals,
            "matches": matches,
            "accuracy": round(matches / totals, 4) if totals else None,
        },
        "per_contract": per_contract,
    }


def load_existing_eval_outputs(output_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    contracts = select_test_contracts(limit=3)
    pairs: list[tuple[Path, dict[str, Any]]] = []
    for index, pdf in enumerate(contracts, start=1):
        path = output_dir / f"contract_{index}.json"
        if not path.exists():
            raise FileNotFoundError(f"Sortie agent manquante: {path}")
        pairs.append((pdf, json.loads(path.read_text(encoding="utf-8"))))
    return pairs

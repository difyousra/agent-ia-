from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

from .dataset import download_atticus_dataset


YES_NO_CATEGORIES = {
    "Most Favored Nation",
    "Non-Compete",
    "Exclusivity",
    "No-Solicit of Customers",
    "No-Solicit of Employees",
    "Non-Disparagement",
    "Termination for Convenience",
    "ROFR/ROFO/ROFN",
    "Change of Control",
    "Anti-assignment",
    "Revenue/Profit Sharing",
    "Price Restrictions",
    "Minimum Commitment",
    "Volume Restriction",
    "IP Ownership Assignment",
    "Joint IP Ownership",
    "License Grant",
    "Non-Transferable License",
    "Affiliate License-Licensor",
    "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License",
    "Source Code Escrow",
    "Post-Termination Services",
    "Audit Rights",
    "Uncapped Liability",
    "Cap On Liability",
    "Liquidated Damages",
    "Warranty Duration",
    "Insurance",
    "Covenant Not To Sue",
    "Third Party Beneficiary",
}

EXTRACTIVE_CATEGORIES = {
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Governing Law",
}


def find_clause_csv(contract_pdf: Path, dataset_root: Path | None = None) -> Path:
    root = dataset_root or download_atticus_dataset()
    clauses_dir = root / "Final Publication" / "individual_contract_clauses"
    expected = clauses_dir / f"{contract_pdf.stem}.csv"
    if expected.exists():
        return expected
    matches = [
        path
        for path in clauses_dir.glob("*.csv")
        if path.stem.lower() == contract_pdf.stem.lower()
    ]
    if not matches:
        raise FileNotFoundError(f"Annotation CSV introuvable pour {contract_pdf.name}")
    return matches[0]


def load_atticus_labels(csv_path: Path) -> dict[str, dict]:
    """Charge les labels Atticus pour un contrat.

    Retourne {category: {"answers": [...], "clauses": [...], "present": bool}}
    """
    buckets: dict[str, dict] = defaultdict(lambda: {"answers": [], "clauses": []})
    with csv_path.open(encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            clause = (row.get("Clause") or "").strip()
            for index in range(1, 8):
                label = (row.get(f"Label {index}") or "").strip()
                answer = (row.get(f"Label {index}-Answer") or "").strip()
                if not label:
                    continue
                buckets[label]["answers"].append(answer)
                if clause:
                    buckets[label]["clauses"].append(clause)

    result: dict[str, dict] = {}
    for label, payload in buckets.items():
        answers = [a for a in payload["answers"] if a]
        primary = answers[0] if answers else None
        if label in YES_NO_CATEGORIES:
            present = any(a.lower() == "yes" for a in answers) or bool(answers)
            if answers and all(a.lower() == "no" for a in answers):
                present = False
        else:
            present = bool(primary)
        result[label] = {
            "answers": answers,
            "answer": primary,
            "clauses": payload["clauses"],
            "present": present,
        }
    return result


def normalize_date(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    match = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if match:
        month, day, year = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    return text


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    text = value.lower()
    text = re.sub(r"[\"'`]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def party_names_from_atticus(answer: str | None) -> list[str]:
    if not answer:
        return []
    parts = re.split(r";|\n", answer)
    names: list[str] = []
    for part in parts:
        cleaned = re.sub(r"\([^)]*\)", "", part).strip(" ,")
        cleaned = re.sub(r'\s*"\w+"\s*$', "", cleaned).strip(" ,")
        if cleaned:
            names.append(cleaned)
    return names

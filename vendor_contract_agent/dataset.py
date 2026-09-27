from __future__ import annotations

from pathlib import Path

import kagglehub

ATTICUS_DATASET = "theatticusproject/atticus-open-contract-dataset-aok-beta/versions/1"

SELECTED_CONTRACTS = [
    "PcquoteComInc_19990721_S-1A_EX-10.11_6377149_EX-10.11_Co-Branding Agreement2.pdf",
    "ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT VENTURE AGREEMENT.PDF",
    "DataCallTechnologies_20060918_SB-2A_EX-10.9_944510_EX-10.9_Content License Agreement.pdf",
]


def download_atticus_dataset() -> Path:
    return Path(kagglehub.dataset_download(ATTICUS_DATASET))


def select_test_contracts(root: Path | None = None, limit: int = 3) -> list[Path]:
    dataset_root = root or download_atticus_dataset()
    pdfs = {path.name: path for path in dataset_root.rglob("*.pdf")}
    selected: list[Path] = []
    for name in SELECTED_CONTRACTS[:limit]:
        if name not in pdfs:
            raise FileNotFoundError(f"Contrat introuvable dans le dataset: {name}")
        selected.append(pdfs[name])
    return selected

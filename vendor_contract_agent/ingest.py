from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".docx"}


def extract_contract_text(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Format non supporté: {suffix}. Formats acceptés: {sorted(SUPPORTED_SUFFIXES)}"
        )

    if suffix == ".pdf":
        return _extract_pdf(file_path)
    if suffix == ".docx":
        return _extract_docx(file_path)
    return file_path.read_text(encoding="utf-8")


def _extract_pdf(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"--- PAGE {index} ---\n{text}")
    combined = "\n\n".join(pages).strip()
    if not combined:
        raise ValueError("Aucun texte extractible dans le PDF.")
    return combined


def _extract_docx(file_path: Path) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("python-docx est requis pour lire les fichiers .docx") from exc

    document = Document(str(file_path))
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text and p.text.strip()]
    combined = "\n".join(paragraphs).strip()
    if not combined:
        raise ValueError("Aucun texte extractible dans le DOCX.")
    return combined

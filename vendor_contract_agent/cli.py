from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from .agent import VendorContractAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyse un contrat fournisseur et produit un JSON actionnable."
    )
    parser.add_argument(
        "contract",
        nargs="?",
        help="Chemin du contrat (.pdf, .txt, .md, .docx)",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Lancer l'interface web Streamlit",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Lancer l'API FastAPI RAG (uvicorn)",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="Port de l'API FastAPI (défaut: 8000)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="outputs.json",
        help="Fichier JSON de sortie (défaut: outputs.json)",
    )
    parser.add_argument(
        "--print",
        dest="print_json",
        action="store_true",
        help="Afficher le JSON sur la sortie standard",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv(override=True)
    args = build_parser().parse_args(argv)
    if args.ui:
        return _launch_ui()
    if args.api:
        return _launch_api(args.api_port)
    if not args.contract:
        print("Erreur: indiquez un fichier contrat, ou utilisez --ui / --api.", file=sys.stderr)
        return 2
    contract_path = Path(args.contract)
    output_path = Path(args.output)

    try:
        result = VendorContractAgent().analyze_file(contract_path)
    except Exception as exc:
        print(f"Erreur: {exc}", file=sys.stderr)
        return 1

    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.print_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Analyse écrite dans {output_path}")
    return 0


def _launch_ui() -> int:
    import subprocess

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.headless", "true"]
    return subprocess.call(command)


def _launch_api(port: int) -> int:
    import subprocess

    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "vendor_contract_agent.api:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
    ]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())

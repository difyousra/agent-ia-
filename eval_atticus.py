from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from vendor_contract_agent.agent import VendorContractAgent
from vendor_contract_agent.dataset import select_test_contracts


def main() -> int:
    load_dotenv(override=True)
    output_dir = Path("eval_atticus")
    output_dir.mkdir(exist_ok=True)

    contracts = select_test_contracts(limit=3)
    agent = VendorContractAgent()
    summary = []

    for index, contract_path in enumerate(contracts, start=1):
        print(f"[{index}/3] Analyse: {contract_path.name}")
        try:
            result = agent.analyze_file(contract_path)
            out_file = output_dir / f"contract_{index}.json"
            out_file.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            contract = result.get("contract") or {}
            parties = result.get("parties") or {}
            vendor = (parties.get("vendor") or {}).get("name")
            customer = (parties.get("customer") or {}).get("name")
            summary.append(
                {
                    "file": contract_path.name,
                    "ok": True,
                    "output": str(out_file),
                    "contract_number": contract.get("contract_number"),
                    "title": contract.get("contract_title"),
                    "vendor": vendor,
                    "customer": customer,
                    "actions": len(result.get("actions") or []),
                    "risks": len(result.get("risks") or []),
                    "error": None,
                }
            )
            print(
                f"  OK  {contract.get('contract_title') or contract.get('contract_type')} | "
                f"actions={len(result.get('actions') or [])} risques={len(result.get('risks') or [])}"
            )
        except Exception as exc:
            summary.append(
                {
                    "file": contract_path.name,
                    "ok": False,
                    "output": None,
                    "error": str(exc),
                }
            )
            print(f"  ECHEC  {exc}")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Resume: {summary_path}")

    if all(item.get("ok") for item in summary):
        from vendor_contract_agent.evaluate import evaluate_selected, load_existing_eval_outputs

        report = evaluate_selected(load_existing_eval_outputs(output_dir))
        report_path = output_dir / "evaluation_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        overall = report["overall"]
        print(
            f"Evaluation Atticus: {overall['matches']}/{overall['total']} "
            f"(accuracy={overall['accuracy']}) -> {report_path}"
        )

    return 0 if all(item.get("ok") for item in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())

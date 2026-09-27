from __future__ import annotations

import json
from pathlib import Path

from vendor_contract_agent.evaluate import evaluate_selected, load_existing_eval_outputs


def _print_report(report: dict) -> None:
    overall = report["overall"]
    print("=== Evaluation Atticus vs Agent ===")
    print(
        f"Contrats: {report['contracts_evaluated']} | "
        f"matches={overall['matches']}/{overall['total']} | "
        f"accuracy={overall['accuracy']}"
    )
    for item in report["per_contract"]:
        metrics = item["metrics"]
        print(
            f"\n[{item['file']}]\n"
            f"  accuracy={metrics['accuracy']} "
            f"({metrics['matches']}/{metrics['total']}) | "
            f"extractive={metrics['extractive_matches']}/{metrics['extractive_total']} | "
            f"presence={metrics['presence_matches']}/{metrics['presence_total']}"
        )
        for comparison in item["comparisons"]:
            mark = "OK" if comparison["match"] else "KO"
            print(
                f"  [{mark}] {comparison['category']}: "
                f"GT={comparison.get('ground_truth')!r} | "
                f"Agent={comparison.get('agent')!r}"
            )


def main() -> int:
    output_dir = Path("eval_atticus")
    pairs = load_existing_eval_outputs(output_dir)
    report = evaluate_selected(pairs)
    out = output_dir / "evaluation_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print_report(report)
    print(f"\nRapport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

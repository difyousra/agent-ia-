import json
from pathlib import Path

from vendor_contract_agent.ingest import extract_contract_text
from vendor_contract_agent.normalize import extract_json_payload, normalize_list_items, normalize_result


def test_extract_json_from_fenced_block():
    raw = """```json
    {"contract": {"contract_number": "VSA-1"}}
    ```"""
    data = extract_json_payload(raw)
    assert data["contract"]["contract_number"] == "VSA-1"


def test_normalize_fills_missing_keys_and_drops_empty_penalties():
    parsed = {
        "contract": {"contract_number": "VSA-2026-047"},
        "penalties": [{"type": None, "trigger": None, "amount": None, "percentage": None, "description": None}],
        "actions": [
            {
                "action": "Submit monthly service report",
                "responsible_party": "vendor",
                "deadline": "5th business day",
                "frequency": "monthly",
                "priority": "medium",
                "source_clause": "6",
            }
        ],
    }
    result = normalize_list_items(normalize_result(parsed))
    assert result["contract"]["contract_number"] == "VSA-2026-047"
    assert result["contract"]["start_date"] is None
    assert result["parties"]["vendor"]["name"] is None
    assert result["scope"]["products"] == []
    assert result["penalties"] == []
    assert result["actions"][0]["action"] == "Submit monthly service report"
    assert result["actions"][0]["frequency"] == "monthly"


def test_prompt_placeholder_survives_braces_in_contract():
    from vendor_contract_agent.prompts import USER_PROMPT_TEMPLATE

    text = "Fee is {12%} of {revenue}."
    prompt = USER_PROMPT_TEMPLATE.replace("<<<CONTRACT_TEXT>>>", text)
    assert "Fee is {12%} of {revenue}." in prompt

    pdf = Path(__file__).resolve().parents[1] / "sample_vendor_contract_test.pdf"
    text = extract_contract_text(pdf)
    assert "VSA-2026-047" in text
    assert "Atlas Digital Solutions" in text

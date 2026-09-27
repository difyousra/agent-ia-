from vendor_contract_agent.atticus_gt import normalize_date, party_names_from_atticus
from vendor_contract_agent.evaluate import _names_match, _renewal_match, compare_category


def test_normalize_date():
    assert normalize_date("03/01/2003") == "2003-03-01"
    assert normalize_date("2003-03-01") == "2003-03-01"


def test_party_match():
    gt = party_names_from_atticus('PC Quote, Inc. ("PCQ"); A.B. Watley, Inc. ("ABW")')
    assert _names_match(gt, ["PC Quote, Inc.", "A.B. Watley, Inc."])


def test_renewal_alias():
    assert _renewal_match("0.5 year", "half a year")


def test_presence_compare():
    result = compare_category(
        "Audit Rights",
        {"answer": "Yes", "present": True},
        {"answer": "Yes", "present": True},
    )
    assert result["match"] is True

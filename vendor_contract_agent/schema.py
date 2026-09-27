EMPTY_SCHEMA = {
    "contract": {
        "contract_number": None,
        "contract_title": None,
        "contract_type": None,
        "effective_date": None,
        "start_date": None,
        "end_date": None,
        "status": None,
    },
    "parties": {
        "vendor": {
            "name": None,
            "address": None,
            "registration_number": None,
            "contact": None,
        },
        "customer": {
            "name": None,
            "address": None,
            "registration_number": None,
            "contact": None,
        },
    },
    "scope": {
        "description": None,
        "products": [],
        "services": [],
        "deliverables": [],
    },
    "financial": {
        "contract_value": None,
        "currency": None,
        "taxes": None,
        "pricing_model": None,
        "discounts": [],
        "additional_fees": [],
    },
    "payment": {
        "payment_method": None,
        "payment_terms": None,
        "payment_due_days": None,
        "payment_frequency": None,
        "invoice_requirements": [],
    },
    "delivery": {
        "delivery_terms": None,
        "delivery_location": None,
        "delivery_deadlines": [],
        "milestones": [],
    },
    "sla": {
        "applicable": None,
        "availability": None,
        "response_time": None,
        "resolution_time": None,
        "support_hours": None,
        "service_credits": [],
    },
    "obligations": {
        "vendor": [],
        "customer": [],
    },
    "warranty": {
        "applicable": None,
        "period": None,
        "coverage": None,
        "exclusions": [],
    },
    "support_and_maintenance": {
        "support_included": None,
        "maintenance_included": None,
        "support_period": None,
        "support_channels": [],
    },
    "intellectual_property": {
        "ownership": None,
        "license": None,
        "restrictions": [],
    },
    "confidentiality": {
        "applicable": None,
        "duration": None,
        "requirements": [],
    },
    "data_protection": {
        "personal_data": None,
        "data_types": [],
        "retention_period": None,
        "data_location": None,
        "breach_notification": None,
    },
    "security": {
        "requirements": [],
        "certifications": [],
        "audit_requirements": [],
    },
    "compliance": {
        "laws": [],
        "regulations": [],
        "standards": [],
        "certifications": [],
    },
    "insurance": {
        "required": None,
        "types": [],
        "coverage": None,
    },
    "liability": {
        "liability_cap": None,
        "currency": None,
        "excluded_damages": [],
        "uncapped_liabilities": [],
    },
    "penalties": [],
    "force_majeure": {
        "applicable": None,
        "events": [],
        "notification_period": None,
    },
    "subcontracting": {
        "allowed": None,
        "approval_required": None,
        "conditions": [],
    },
    "audit": {
        "allowed": None,
        "frequency": None,
        "notice_period": None,
    },
    "termination": {
        "for_convenience": None,
        "for_cause": None,
        "notice_period_days": None,
        "termination_events": [],
        "consequences": [],
    },
    "renewal": {
        "automatic": None,
        "renewal_period": None,
        "notice_period_days": None,
        "maximum_renewals": None,
    },
    "dispute_resolution": {
        "method": None,
        "jurisdiction": None,
        "arbitration": None,
    },
    "governing_law": {
        "country": None,
        "law": None,
    },
    "important_dates": [],
    "actions": [],
    "risks": [],
    "missing_information": [],
    "sources": [],
}

PENALTY_ITEM = {
    "type": None,
    "trigger": None,
    "amount": None,
    "percentage": None,
    "description": None,
    "source_clause": None,
}

DATE_ITEM = {
    "date": None,
    "event": None,
    "description": None,
    "source_clause": None,
}

ACTION_ITEM = {
    "action": None,
    "responsible_party": None,
    "deadline": None,
    "frequency": None,
    "priority": None,
    "source_clause": None,
}

RISK_ITEM = {
    "risk": None,
    "severity": None,
    "trigger": None,
    "consequence": None,
    "source_clause": None,
}

SOURCE_ITEM = {
    "clause": None,
    "page": None,
    "text": None,
}

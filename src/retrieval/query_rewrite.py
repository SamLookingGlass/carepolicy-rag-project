"""Expand acronyms before retrieval so keyword search doesn't miss them.

Example: "What is CHAS?" also searches "Community Health Assist Scheme",
because the source pages often spell the name out instead of using the acronym.
"""

from __future__ import annotations

import re

# Acronym → expanded terms appended for hybrid retrieval
EXPANSIONS: dict[str, str] = {
    r"\bchas\b": "Community Health Assist Scheme CHAS",
    r"\bcdmp\b": "Chronic Disease Management Programme CDMP",
    r"\bpdpa\b": "Personal Data Protection Act PDPA",
    r"\bmoh\b": "Ministry of Health MOH",
    r"\bhpb\b": "Health Promotion Board HPB",
    r"\bfit\b": "Faecal Immunochemical Test FIT colorectal screening",
    r"\bmedishield\b": "MediShield Life insurance",
    r"\bmedisave\b": "MediSave CPF medical savings",
    r"\bsfl\b": "Screen for Life SFL",
    r"\bnehr\b": "National Electronic Health Record NEHR",
    r"\bdpo\b": "Data Protection Officer DPO",
}


def rewrite_query(query: str) -> str:
    """Append expanded terms for acronyms found in the query."""
    expanded_parts: list[str] = [query]
    lower = query.lower()

    for pattern, expansion in EXPANSIONS.items():
        if re.search(pattern, lower, re.IGNORECASE):
            expanded_parts.append(expansion)

    return " ".join(expanded_parts)

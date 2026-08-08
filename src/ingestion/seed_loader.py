"""Offline seed documents when live fetch is unavailable."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import RAW_DIR

from src.ingestion.seed_expansion import EXTRA_SEED_DOCUMENTS

# Seed documents are hand-written summaries used when live fetch is unavailable
# or returns too few pages (government sites restructure URLs frequently).
#
# Seed docs carry no per-topic URL; load_seed_documents() cites each source's
# verified homepage instead of a deep link that could rot or never existed.
HOMEPAGE_BY_SOURCE: dict[str, str] = {
    "healthhub": "https://www.healthhub.sg/",
    "moh": "https://www.moh.gov.sg/",
    "hpb": "https://www.hpb.gov.sg/",
    "pdpc": "https://www.pdpc.gov.sg/",
}


def _verified_citation(source: str) -> tuple[str, str]:
    """Return (url, domain) pointing to a real, verified page for this source."""
    url = HOMEPAGE_BY_SOURCE.get(source, "https://www.gov.sg/")
    domain = url.split("//", 1)[1].rstrip("/")
    return url, domain


SEED_DOCUMENTS: list[dict[str, str]] = [
    {
        "doc_id": "seed_chas",
        "source": "healthhub",
        "category": "subsidies",
        "title": "Community Health Assist Scheme (CHAS)",
        "text": """Community Health Assist Scheme (CHAS)

CHAS enables all Singapore Citizens, including Pioneer Generation (PG) and Merdeka Generation (MG) cardholders, to receive subsidies for medical and dental care at participating general practitioner (GP) and dental clinics.

CHAS covers chronic conditions including diabetes, high blood pressure, high cholesterol, stroke, asthma, COPD, depression, anxiety, dementia, osteoarthritis, rheumatoid arthritis, and selected other conditions.

Orange CHAS cardholders receive subsidies for common illnesses and selected dental services. Blue CHAS cardholders receive higher subsidies. Green CHAS cardholders receive subsidies for chronic conditions under the Chronic Disease Management Programme (CDMP).

Eligibility is based on household monthly income per person or Annual Value (AV) of home. Singapore Citizens may apply through the HealthHub website or at clinics displaying the CHAS decal.""",
    },
    {
        "doc_id": "seed_medishield",
        "source": "healthhub",
        "category": "subsidies",
        "title": "MediShield Life",
        "text": """MediShield Life

MediShield Life is a basic health insurance plan administered by the Central Provident Fund (CPF) Board that helps pay for large hospital bills and selected costly outpatient treatments for all Singapore Citizens and Permanent Residents.

It provides lifelong protection regardless of age or pre-existing conditions. Premiums can be paid using MediSave. Claim limits apply to policy year and lifetime amounts.

MediShield Life covers inpatient hospitalisation, day surgery, and selected outpatient treatments such as chemotherapy, radiotherapy, kidney dialysis, and erythropoietin for chronic kidney failure.

Policyholders may purchase Integrated Shield Plans (IPs) from private insurers for additional coverage and access to private hospitals, subject to insurer underwriting for new riders.""",
    },
    {
        "doc_id": "seed_medisave",
        "source": "healthhub",
        "category": "subsidies",
        "title": "MediSave",
        "text": """MediSave

MediSave is a national medical savings scheme that helps CPF members save for future healthcare expenses. CPF members contribute a portion of monthly wages to MediSave.

MediSave can be used to pay for hospitalisation, day surgery, approved outpatient treatments, health insurance premiums including MediShield Life, and selected preventive care under Flexi-MediSave for eligible seniors.

Withdrawal limits apply for different types of treatments. MediSave can also be used for approved dependents including spouse, children, parents, and grandparents subject to prevailing limits.""",
    },
    {
        "doc_id": "seed_screen_for_life",
        "source": "healthhub",
        "category": "screening",
        "title": "Screen for Life",
        "text": """Screen for Life

Screen for Life (SFL) is a national screening programme encouraging Singapore Citizens and Permanent Residents to go for regular health screening and follow-up.

Eligible Singapore Citizens receive subsidised screening at CHAS GP clinics and polyclinics. Pioneer Generation and Merdeka Generation members receive higher subsidies.

Recommended screenings include obesity (BMI), diabetes (HbA1c or fasting glucose), hypertension, hyperlipidaemia, colorectal cancer (FIT), cervical cancer (Pap smear or HPV DNA test for eligible women), and breast cancer (mammogram for women aged 50 to 69).

Screening frequency depends on age and risk factors. Healthier SG enrollees may receive additional subsidies for recommended screenings.""",
    },
    {
        "doc_id": "seed_diabetes",
        "source": "healthhub",
        "category": "prevention",
        "title": "Diabetes Prevention and Management",
        "text": """Diabetes

Diabetes is a condition where blood glucose levels are too high. Type 2 diabetes is the most common form in Singapore and is linked to obesity, physical inactivity, and family history.

Preventive measures include maintaining healthy weight, regular physical activity, balanced diet low in sugar and refined carbohydrates, and regular screening for those at risk.

The Chronic Disease Management Programme (CDMP) provides subsidised outpatient care for diabetes at GP clinics and polyclinics for patients with CHAS, PG, MG, or Merdeka cards.

Complications of poorly controlled diabetes include kidney disease, eye disease, nerve damage, and cardiovascular disease. Patients should monitor blood glucose as advised and attend regular follow-up.""",
    },
    {
        "doc_id": "seed_hypertension",
        "source": "healthhub",
        "category": "prevention",
        "title": "High Blood Pressure",
        "text": """High Blood Pressure (Hypertension)

Hypertension is defined as sustained blood pressure readings of 140/90 mmHg or higher. It is a major risk factor for stroke, heart attack, and kidney failure.

Lifestyle modifications include reducing salt intake, maintaining healthy weight, regular exercise, limiting alcohol, and smoking cessation.

Hypertension is covered under CDMP with subsidised GP and polyclinic visits for eligible patients. Regular monitoring and medication adherence are important for control.""",
    },
    {
        "doc_id": "seed_pdpa_overview",
        "source": "pdpc",
        "category": "compliance",
        "title": "Personal Data Protection Act (PDPA)",
        "text": """Personal Data Protection Act (PDPA)

The PDPA governs the collection, use, and disclosure of personal data by organisations in Singapore. It establishes a baseline standard for personal data protection.

Key obligations include consent, purpose limitation, notification, access and correction, accuracy, protection, retention limitation, transfer limitation, and openness.

Organisations must appoint at least one Data Protection Officer (DPO). The PDPA applies to healthcare organisations handling patient personal data.

Breaches of protection obligations may require notification to PDPC and affected individuals under the mandatory data breach notification regime.""",
    },
    {
        "doc_id": "seed_pdpa_healthcare",
        "source": "pdpc",
        "category": "compliance",
        "title": "Data Protection for Healthcare",
        "text": """Data Protection for Healthcare Sector

Healthcare organisations collect sensitive personal data including medical records, NRIC, contact details, and insurance information. They must comply with PDPA obligations.

Patient data may be collected for treatment, payment, and healthcare operations. Consent may be deemed in certain circumstances for purposes clearly in the patient's interest.

Sharing patient data between healthcare institutions requires appropriate legal basis such as consent or specific exceptions under the PDPA. Safeguards include access controls, encryption, and staff training.

Healthcare providers should implement data minimisation, secure systems, and audit trails when sharing data for continuity of care.""",
    },
    {
        "doc_id": "seed_pdpa_consent",
        "source": "pdpc",
        "category": "compliance",
        "title": "PDPA Consent Obligation",
        "text": """Consent Obligation under PDPA

Organisations must obtain consent before collecting, using, or disclosing personal data unless an exception applies. Consent must be informed and voluntarily given.

Deemed consent may apply when individuals voluntarily provide personal data for a reasonable purpose. Consent may be withdrawn with reasonable notice.

Healthcare organisations should document consent for non-routine uses of patient data and provide clear notices about data purposes at collection points.""",
    },
    {
        "doc_id": "seed_polyclinic",
        "source": "healthhub",
        "category": "policy",
        "title": "Polyclinics in Singapore",
        "text": """Polyclinics

Polyclinics are subsidised primary care clinics operated by SingHealth, NHG, and NUHS serving as accessible healthcare touchpoints for Singaporeans and PRs.

Services include outpatient consultations, vaccinations, health screening, chronic disease management, and referrals to hospitals. Subsidies apply for eligible citizens and PRs.

Polyclinics support Screen for Life, CDMP, and national vaccination programmes. Appointment booking is available through HealthHub.""",
    },
    {
        "doc_id": "seed_gp",
        "source": "healthhub",
        "category": "policy",
        "title": "General Practitioner (GP) Clinics",
        "text": """General Practitioner Clinics

GP clinics provide primary care including treatment of common illnesses, chronic disease management, vaccinations, and health screening. CHAS-accredited clinics display the CHAS decal.

CHAS, PG, and MG cardholders receive subsidies at participating GP clinics. GPs play a key role in the Chronic Disease Management Programme and Healthier SG enrolment.""",
    },
    {
        "doc_id": "seed_colorectal",
        "source": "healthhub",
        "category": "screening",
        "title": "Colorectal Cancer Screening",
        "text": """Colorectal Cancer Screening

Colorectal cancer is among the most common cancers in Singapore. Screening helps detect cancer early when treatment is more effective.

The Faecal Immunochemical Test (FIT) is recommended for average-risk individuals aged 50 and above. Screen for Life provides subsidised FIT screening for eligible Singaporeans.

Positive FIT results require follow-up colonoscopy. Screening frequency is typically once a year for FIT unless otherwise advised.""",
    },
    {
        "doc_id": "seed_cervical",
        "source": "healthhub",
        "category": "screening",
        "title": "Cervical Cancer Screening",
        "text": """Cervical Cancer Screening

Cervical cancer screening is recommended for women aged 25 to 29 with Pap smear every 3 years, and women aged 30 to 69 with HPV DNA test every 5 years.

Screen for Life provides subsidised screening at CHAS GP clinics and polyclinics. HPV vaccination is recommended for females aged 9 to 26 as primary prevention.""",
    },
    {
        "doc_id": "seed_breast",
        "source": "healthhub",
        "category": "screening",
        "title": "Breast Cancer Screening",
        "text": """Breast Cancer Screening

Mammogram screening is recommended every 2 years for women aged 50 to 69 with no symptoms. Screen for Life offers subsidised mammograms at approved centres for eligible Singaporeans.

Women with family history or other risk factors may need earlier or more frequent screening as advised by their doctor.""",
    },
    {
        "doc_id": "seed_hpb_screenings",
        "source": "hpb",
        "category": "screening",
        "title": "HPB Health Screenings",
        "text": """Health Promotion Board — Screenings

HPB promotes national health screening through Screen for Life and community outreach. Regular screening detects chronic conditions and cancers early.

Key messages include knowing your screening eligibility by age and gender, using subsidised screening at primary care providers, and following up on abnormal results promptly.""",
    },
]


def _wrap_html(title: str, text: str) -> str:
    paragraphs = "".join(f"<p>{p.strip()}</p>" for p in text.strip().split("\n\n") if p.strip())
    return f"<!DOCTYPE html><html><head><title>{title}</title></head><body><main>{paragraphs}</main></body></html>"


def all_seed_documents() -> list[dict[str, str]]:
    return SEED_DOCUMENTS + EXTRA_SEED_DOCUMENTS


def load_seed_documents(raw_dir: Path = RAW_DIR) -> list[dict]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    loaded: list[dict] = []

    for doc in all_seed_documents():
        # Cite the source's verified homepage; seed docs carry no per-topic URL.
        verified_url, verified_domain = _verified_citation(doc["source"])
        doc = {**doc, "url": verified_url, "domain": verified_domain}

        html_path = raw_dir / f"{doc['doc_id']}.html"
        meta_path = raw_dir / f"{doc['doc_id']}.meta.json"

        html_path.write_text(_wrap_html(doc["title"], doc["text"]), encoding="utf-8")
        meta_path.write_text(
            json.dumps(
                {
                    "doc_id": doc["doc_id"],
                    "url": doc["url"],
                    "source": doc["source"],
                    "category": doc["category"],
                    "title": doc["title"],
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "domain": doc["domain"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        loaded.append(doc)

    return loaded

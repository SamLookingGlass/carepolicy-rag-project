"""Build golden eval set aligned to the live scraped corpus.

Run: python scripts/build_golden_live.py
Then validate: python scripts/validate_golden.py
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import EVAL_DIR

# (id, question, domains, reference_answer, category, expect_refusal)
LIVE_GOLDEN: list[tuple] = [
    # subsidies
    ("q001", "What is CHAS and who is eligible?", ["healthhub.sg", "moh.gov.sg"],
     "CHAS provides subsidised medical and dental care at participating GP and dental clinics for all Singapore Citizens, tiered by income or Annual Value.", "subsidies", False),
    ("q002", "What chronic conditions are covered under CHAS?", ["healthhub.sg", "moh.gov.sg"],
     "CHAS covers selected chronic conditions under the CHAS Chronic Tier with subsidised GP visits.", "subsidies", False),
    ("q003", "What is MediShield Life and who is covered?", ["healthhub.sg", "moh.gov.sg"],
     "MediShield Life is basic health insurance covering all Singapore Citizens and Permanent Residents for life.", "subsidies", False),
    ("q006", "What is MediFund and who can approach it?", ["healthhub.sg"],
     "MediFund helps lower-income Singaporeans with remaining healthcare bills after subsidies, MediShield Life, IPs, MediSave, and cash.", "subsidies", False),
    ("q008", "Can MediSave be used for dependents?", ["healthhub.sg", "moh.gov.sg"],
     "MediSave can be withdrawn for the account holder and immediate family members subject to withdrawal limits.", "subsidies", False),
    ("q016", "What is the Chronic Disease Management Programme?", ["healthhub.sg", "moh.gov.sg"],
     "CDMP provides subsidised outpatient care for chronic conditions at GP clinics and polyclinics.", "subsidies", False),
    ("q017", "What is CareShield Life?", ["healthhub.sg", "moh.gov.sg"],
     "CareShield Life is long-term care insurance providing basic support when severely disabled.", "subsidies", False),
    ("q018", "What benefits does the Merdeka Generation Package provide?", ["moh.gov.sg"],
     "Merdeka Generation cardholders receive enhanced healthcare subsidies similar to other generation packages.", "subsidies", False),
    ("q019", "What subsidies do Pioneer Generation members receive?", ["healthhub.sg", "moh.gov.sg"],
     "Pioneers receive MediSave top-ups, additional outpatient subsidies, CareShield incentives, and MediShield premium subsidies.", "subsidies", False),
    ("q056", "What are CHAS Blue card subsidies for common illness?", ["healthhub.sg", "moh.gov.sg"],
     "CHAS Blue cardholders receive higher subsidies than Orange or Green cardholders for common illnesses.", "subsidies", False),
    ("q057", "Can MediShield Life premiums be paid with MediSave?", ["healthhub.sg", "moh.gov.sg"],
     "Yes, MediShield Life premiums can be paid using MediSave.", "subsidies", False),
    ("q067", "What is MediSave used for besides hospitalisation?", ["healthhub.sg", "moh.gov.sg"],
     "MediSave covers approved outpatient treatments, insurance premiums, and other approved healthcare expenses.", "subsidies", False),
    ("q068", "What is an Integrated Shield Plan?", ["healthhub.sg", "moh.gov.sg"],
     "IPs add private insurance coverage on top of the MediShield Life component.", "subsidies", False),
    ("q074", "What subsidies are available for inpatient care at public hospitals?", ["moh.gov.sg"],
     "Singapore Citizens receive tiered inpatient subsidies at public healthcare institutions based on means testing.", "subsidies", False),
    # screening
    ("q004", "What conditions does Healthier SG Screening cover?", ["healthhub.sg", "moh.gov.sg"],
     "Healthier SG Screening covers cardiovascular risk factors such as obesity, diabetes, hypertension, and hyperlipidaemia, plus cervical cancer screening.", "screening", False),
    ("q005", "Who is eligible for cervical cancer screening under Healthier SG Screening?", ["healthhub.sg"],
     "Female Singapore Citizens aged 25 and above who have had sexual activity and are due for Pap or HPV screening.", "screening", False),
    ("q027", "How often is cervical cancer screening recommended under Healthier SG?", ["healthhub.sg"],
     "Pap test every three years or HPV test every five years depending on age and prior results.", "screening", False),
    ("q028", "What is the Diabetes Risk Assessment tool for adults aged 18 to 39?", ["healthhub.sg"],
     "DRA is a self-administered tool for ages 18-39 to identify higher risk of undiagnosed type 2 diabetes.", "screening", False),
    ("q030", "How much does Healthier SG Screening cost for Singapore Citizens?", ["healthhub.sg"],
     "Eligible Singapore Citizens pay $5 or less per screening visit with the Healthier SG Screening subsidy.", "screening", False),
    ("q071", "Where can I go for Healthier SG Screening?", ["healthhub.sg"],
     "Healthier SG Screening is available at participating CHAS GP clinics islandwide.", "screening", False),
    # prevention
    ("q012", "What lifestyle changes help prevent type 2 diabetes?", ["healthhub.sg"],
     "Prevention includes healthy weight, regular physical activity, balanced diet, and regular screening.", "prevention", False),
    ("q022", "What is the National Adult Immunisation Schedule?", ["moh.gov.sg"],
     "NAIS provides guidance on vaccinations that adults aged 18 and older should receive.", "prevention", False),
    ("q023", "How does HPB support smoking cessation?", ["hpb.gov.sg"],
     "HPB offers smoking cessation e-learning for professionals and notes that brief physician advice increases quit attempts.", "prevention", False),
    ("q024", "What physical activity programmes does HPB run?", ["hpb.gov.sg"],
     "HPB runs MOVE IT programmes at community centres, parks, malls, and Active Ageing Centres for all ages.", "prevention", False),
    ("q025", "Why is cardiovascular risk screening important?", ["healthhub.sg"],
     "Conditions like hypertension and diabetes can be silent early on; screening enables early detection and management.", "prevention", False),
    ("q026", "What is stroke prevention advice?", ["healthhub.sg"],
     "Manage hypertension, diabetes, cholesterol, quit smoking, exercise, and maintain a healthy diet.", "prevention", False),
    ("q029", "What side effects can occur after an influenza vaccine?", ["healthhub.sg"],
     "Common side effects include soreness at the injection site, headache, tiredness, or fever.", "prevention", False),
    ("q058", "What does CDMP provide for chronic conditions?", ["healthhub.sg", "moh.gov.sg"],
     "CDMP provides subsidised outpatient visits for chronic conditions at GPs and polyclinics.", "prevention", False),
    ("q061", "What is mental well-being according to HealthHub?", ["healthhub.sg"],
     "HealthHub MindSG explains mental well-being, coping with emotions, and self-care tools including a self-assessment.", "prevention", False),
    ("q075", "What is the National Childhood Immunisation Schedule?", ["healthhub.sg"],
     "NCIS defines recommended childhood vaccines; Singaporean children receive subsidised NCIS vaccinations at CHAS GPs and polyclinics.", "prevention", False),
    ("q032", "What NAIS vaccine subsidies are available at public hospitals?", ["moh.gov.sg"],
     "Singapore Citizens can receive up to 75% subsidies for NAIS vaccines at public healthcare institutions.", "prevention", False),
    ("q033", "What vaccines are listed on the HealthHub vaccination clinic page?", ["healthhub.sg"],
     "The page covers HPV, influenza, pneumococcal, Tdap, hepatitis, MMR, and other travel-related vaccines.", "prevention", False),
    # policy
    ("q014", "Where should I go for mild cold and flu symptoms?", ["moh.gov.sg"],
     "Visit a pharmacist, GP clinic, or polyclinic for mild-to-moderate symptoms.", "policy", False),
    ("q020", "How are healthcare subsidies determined for MOH schemes?", ["moh.gov.sg"],
     "Subsidy levels use household means-testing via Monthly Per Capita Household Income or Annual Value.", "policy", False),
    ("q021", "What is the PDPA and who does it apply to?", ["pdpc.gov.sg"],
     "The PDPA sets a baseline standard for protecting personal data in Singapore, governing collection, use, and disclosure by organisations.", "compliance", False),
    ("q039", "What is Advance Care Planning?", ["healthhub.sg"],
     "ACP helps individuals document healthcare preferences and values for future incapacity.", "policy", False),
    ("q040", "When should I visit a hospital emergency department?", ["moh.gov.sg"],
     "Go to the ED only for medical emergencies such as severe pain, breathing difficulty, or uncontrolled bleeding.", "policy", False),
    ("q041", "What subsidies are available for palliative care?", ["moh.gov.sg"],
     "Singapore Citizens can receive up to 80% subsidies for palliative care services based on means testing.", "policy", False),
    ("q042", "How is the PDPA enforced?", ["pdpc.gov.sg"],
     "The PDPC investigates non-compliance and can issue directions and financial penalties under the PDPA.", "compliance", False),
    ("q043", "What is the Human Organ Transplant Act?", ["moh.gov.sg"],
     "HOTA allows recovery of kidneys, heart, liver, and corneas after death unless a person opts out.", "policy", False),
    ("q044", "What long-term care subsidies exist for non-residential services?", ["moh.gov.sg"],
     "MOH subsidises non-residential LTC such as day care, centre-based nursing, and active rehabilitation.", "policy", False),
    ("q069", "What is the Healthier Choice Symbol programme?", ["hpb.gov.sg"],
     "HCS labels healthier packaged food products so consumers can identify better options.", "policy", False),
    ("q070", "What post-stroke rehabilitation options exist?", ["healthhub.sg"],
     "Stroke survivors can access outpatient rehab centres or inpatient rehabilitation units.", "policy", False),
    ("q073", "What are MOH hospital bill fee benchmarks for?", ["moh.gov.sg"],
     "Fee benchmarks help patients compare treatment costs at public and private hospitals.", "policy", False),
    ("q034", "What is an Advance Medical Directive?", ["moh.gov.sg"],
     "An AMD allows a person to register in advance that no extraordinary life-sustaining treatment be used to prolong life.", "policy", False),
    ("q035", "What nursing home services does HealthHub describe?", ["healthhub.sg"],
     "HealthHub describes nursing home admission, subsidies, and support for seniors needing residential care.", "policy", False),
    ("q059", "What specialist outpatient subsidies are available at public hospitals?", ["moh.gov.sg"],
     "Singapore Citizens receive up to 70% subsidies for specialist outpatient clinic visits based on means testing.", "policy", False),
    ("q060", "What is My Healthy Plate?", ["healthhub.sg"],
     "My Healthy Plate is a guide for balanced meals with appropriate portions of different food groups.", "policy", False),
    ("q062", "What is MTERA for organ donation?", ["moh.gov.sg"],
     "MTERA lets adults pledge organs for transplant, therapy, education, or research after death.", "policy", False),
    ("q063", "What financial schemes are listed on HealthHub costs and financing?", ["healthhub.sg"],
     "HealthHub covers MediSave, MediShield Life, CHAS, CareShield Life, IPs, MediFund, and related subsidies.", "policy", False),
    ("q064", "What childhood vaccination subsidies are available?", ["healthhub.sg", "moh.gov.sg"],
     "Singaporean children receive subsidised childhood developmental screenings and vaccinations at CHAS GPs and polyclinics.", "policy", False),
    ("q065", "What is HPB mental well-being support?", ["hpb.gov.sg", "healthhub.sg"],
     "HPB and HealthHub promote mental well-being through programmes, self-assessment tools, and helplines.", "policy", False),
    ("q066", "What is subsidised residential long-term care?", ["moh.gov.sg"],
     "MOH provides means-tested subsidies for residential LTC services such as nursing homes.", "policy", False),
    # synthesis
    ("q015", "How does CDMP help with chronic disease costs?", ["healthhub.sg", "moh.gov.sg"],
     "CDMP provides subsidised outpatient visits for chronic conditions at GPs and polyclinics.", "synthesis", False),
    ("q046", "What does CHAS subsidise at GP clinics?", ["healthhub.sg", "moh.gov.sg"],
     "CHAS subsidises common illnesses, chronic conditions, dental care, and screenings at participating GP clinics.", "synthesis", False),
    ("q047", "How does Healthier SG Screening connect to chronic disease management?", ["healthhub.sg"],
     "Screening detects conditions early; patients with abnormal results can follow up under CDMP with subsidies.", "synthesis", False),
    ("q072", "How does MediSave relate to MediShield Life?", ["healthhub.sg", "moh.gov.sg"],
     "MediSave can pay MediShield Life premiums; MediShield covers large hospital bills.", "synthesis", False),
    ("q007", "What is the difference between HOTA and MTERA?", ["moh.gov.sg"],
     "HOTA mandates organ donation unless opted out; MTERA allows voluntary pledges with next-of-kin consent.", "synthesis", False),
    ("q011", "What does a Data Protection Officer do?", ["pdpc.gov.sg"],
     "A DPO oversees an organisation's PDPA compliance, developing policies and processes for handling personal data.", "compliance", False),
    ("q013", "What is diabetes and why is screening recommended?", ["healthhub.sg"],
     "Diabetes affects how the body processes sugar; early screening helps detect it before complications.", "synthesis", False),
    ("q048", "What activities of daily living does CareShield Life use?", ["moh.gov.sg", "healthhub.sg"],
     "CareShield Life assesses severe disability based on inability to perform at least three of six ADLs.", "synthesis", False),
    ("q036", "What stroke rehabilitation pathways exist after hospital discharge?", ["healthhub.sg"],
     "Patients may attend outpatient rehab several times a week or transfer to an inpatient rehabilitation unit.", "synthesis", False),
    ("q037", "How can patients pay for childhood immunisations?", ["healthhub.sg"],
     "Full subsidies for NCIS vaccines are available for Singaporean children at CHAS GP clinics and polyclinics.", "synthesis", False),
    # refusal: off-topic
    ("q009", "What is the recommended antibiotic regimen for severe pneumonia in ICU patients?", [], "", "refusal", True),
    ("q010", "What is the stock price of Apple Inc today?", [], "", "refusal", True),
    ("q049", "What is the best cryptocurrency to invest in?", [], "", "refusal", True),
    ("q050", "Who will win the next FIFA World Cup?", [], "", "refusal", True),
    ("q051", "What is the surgical procedure for appendectomy?", [], "", "refusal", True),
    ("q052", "What is the weather in Tokyo today?", [], "", "refusal", True),
    ("q053", "What is the capital gains tax rate in Singapore?", [], "", "refusal", True),
    ("q054", "How do I apply for a US visa?", [], "", "refusal", True),
    ("q055", "What is the recommended dosage of paracetamol for a 3-year-old?", [], "", "refusal", True),
    # refusal: not in live corpus
    ("q038", "What is NEHR?", [], "", "refusal", True),
    ("q045", "What is telemedicine in Singapore healthcare?", [], "", "refusal", True),
    # compliance (PDPC pages now indexed via Playwright browser fetch)
    ("q031", "What is the PDPA consent obligation?", ["pdpc.gov.sg"],
     "Organisations must notify individuals and obtain consent before collecting, using, or disclosing personal data.", "compliance", False),
]


def build(output_path: Path | None = None) -> int:
    output_path = output_path or (EVAL_DIR / "golden.jsonl")
    items: list[dict] = []
    seen: set[str] = set()
    for row in LIVE_GOLDEN:
        qid, question, domains, ref, category, expect_refusal = row
        if qid in seen:
            raise ValueError(f"Duplicate question id: {qid}")
        seen.add(qid)
        items.append(
            {
                "id": qid,
                "question": question,
                "expected_source_domains": domains,
                "reference_answer": ref,
                "must_cite": not expect_refusal,
                "category": category,
                "expect_refusal": expect_refusal,
            }
        )

    if len(items) != 75:
        raise ValueError(f"Expected 75 questions, got {len(items)}")

    with output_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    return len(items)


if __name__ == "__main__":
    n = build()
    print(f"Wrote {n} live-corpus golden questions to {EVAL_DIR / 'golden.jsonl'}")

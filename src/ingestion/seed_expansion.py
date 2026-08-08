"""Additional seed documents to reach 80+ corpus size for offline/demo use.

Seed docs carry no url/domain fields; `seed_loader.load_seed_documents()`
assigns each one a verified homepage URL for its source before anything is
written to disk or indexed.
"""

from __future__ import annotations

EXTRA_SEED_DOCUMENTS: list[dict[str, str]] = []

_TOPICS = [
    ("seed_vaccination_childhood", "Childhood Vaccination Schedule", "healthhub", "prevention", "National Childhood Immunisation Schedule covers BCG, Hep B, DTaP, polio, MMR, and other vaccines at recommended ages."),
    ("seed_vaccination_adult", "Adult Vaccination", "healthhub", "prevention", "Adults should stay up to date with influenza, pneumococcal, Tdap boosters, and COVID-19 vaccines as advised by MOH."),
    ("seed_healthier_sg", "Healthier SG", "healthhub", "policy", "Healthier SG is a national initiative encouraging enrolment with a regular family doctor for preventive care and chronic disease management."),
    ("seed_cdmp", "Chronic Disease Management Programme", "healthhub", "subsidies", "CDMP provides subsidised outpatient care for chronic conditions at GP clinics and polyclinics for eligible Singaporeans."),
    ("seed_flexi_medisave", "Flexi-MediSave", "healthhub", "subsidies", "Flexi-MediSave allows eligible seniors to use MediSave for approved outpatient medical services at participating clinics."),
    ("seed_hospital_subsidy", "Hospital Subsidies", "moh", "subsidies", "Subsidised patients receive tiered subsidies at public hospitals based on ward class and means testing."),
    ("seed_means_testing", "Means Testing", "moh", "policy", "Means testing determines patient subsidy levels for healthcare services based on income and property ownership."),
    ("seed_annual_value", "Annual Value for Subsidies", "moh", "policy", "Annual Value of residence is used alongside income to assess eligibility for healthcare subsidies when income data is unavailable."),
    ("seed_pioneer_generation", "Pioneer Generation Package", "healthhub", "subsidies", "Pioneer Generation seniors born before 1950 receive additional healthcare subsidies and MediSave top-ups."),
    ("seed_merdeka_generation", "Merdeka Generation Package", "healthhub", "subsidies", "Merdeka Generation covers Singaporeans born 1950-1959 with enhanced CHAS subsidies and MediSave support."),
    ("seed_mental_health", "Mental Health Support", "hpb", "prevention", "HPB promotes mental wellness through community programmes, helplines, and workplace mental health resources."),
    ("seed_smoking_cessation", "Smoking Cessation", "healthhub", "prevention", "Smoking cessation support includes counselling and nicotine replacement therapies available at polyclinics and selected clinics."),
    ("seed_nutrition", "Healthy Nutrition", "hpb", "prevention", "HPB recommends balanced plates with whole grains, fruits, vegetables, and limited sugar and saturated fat."),
    ("seed_physical_activity", "Physical Activity Guidelines", "hpb", "prevention", "Adults should aim for at least 150 minutes of moderate-intensity aerobic activity per week."),
    ("seed_obesity", "Obesity Management", "healthhub", "prevention", "Obesity increases risk of diabetes and cardiovascular disease; BMI screening is part of Screen for Life."),
    ("seed_stroke", "Stroke Prevention", "healthhub", "prevention", "Stroke risk is reduced by controlling hypertension, diabetes, cholesterol, and avoiding smoking."),
    ("seed_heart_disease", "Heart Disease Prevention", "healthhub", "prevention", "Heart disease prevention includes healthy diet, exercise, and management of blood pressure and cholesterol."),
    ("seed_kidney_disease", "Chronic Kidney Disease", "healthhub", "prevention", "Diabetes and hypertension are leading causes of kidney disease; early detection through screening is important."),
    ("seed_asthma", "Asthma Management", "healthhub", "prevention", "Asthma is a chronic condition covered under CDMP with inhaler therapy and regular follow-up."),
    ("seed_copd", "COPD Management", "healthhub", "prevention", "Chronic obstructive pulmonary disease is often linked to smoking and managed with bronchodilators and pulmonary rehabilitation."),
    ("seed_depression", "Depression", "healthhub", "prevention", "Depression is a mood disorder that may be managed with counselling and medication; seek professional help when symptoms persist."),
    ("seed_anxiety", "Anxiety Disorders", "healthhub", "prevention", "Anxiety disorders cause excessive worry and physical symptoms; treatment includes therapy and medication."),
    ("seed_dementia", "Dementia Care", "healthhub", "prevention", "Dementia affects memory and cognition; early diagnosis helps planning and support for patients and caregivers."),
    ("seed_osteoarthritis", "Osteoarthritis", "healthhub", "prevention", "Osteoarthritis causes joint pain and stiffness; management includes exercise, weight control, and pain relief."),
    ("seed_rheumatoid", "Rheumatoid Arthritis", "healthhub", "prevention", "Rheumatoid arthritis is an autoimmune condition requiring specialist care and disease-modifying drugs."),
    ("seed_palliative", "Palliative Care", "moh", "policy", "Palliative care improves quality of life for patients with serious illness through pain and symptom management."),
    ("seed_end_of_life", "Advance Care Planning", "moh", "policy", "Advance Care Planning allows individuals to document healthcare preferences for future incapacity."),
    ("seed_organ_donation", "Human Organ Transplant Act", "moh", "policy", "HOTA allows organ donation after death for Singapore Citizens and PRs unless they opt out."),
    ("seed_blood_donation", "Blood Donation", "hpb", "prevention", "Regular blood donation is safe for eligible donors and supports national blood supply needs."),
    ("seed_dental_chas", "CHAS Dental Subsidies", "healthhub", "subsidies", "CHAS cardholders receive subsidies for selected dental services at participating clinics."),
    ("seed_eldershield", "CareShield Life", "healthhub", "subsidies", "CareShield Life provides long-term care insurance for severe disability with premiums payable via MediSave."),
    ("seed_community_care", "Community Care Services", "moh", "policy", "Community care includes home care, day care, and nursing homes for seniors needing support."),
    ("seed_intermediate_care", "Intermediate Long-Term Care", "moh", "policy", "Intermediate long-term care facilities provide rehabilitation and sub-acute care after hospitalisation."),
    ("seed_nursing_homes", "Nursing Homes", "moh", "policy", "Nursing homes provide residential care for seniors who require daily assistance and medical supervision."),
    ("seed_home_care", "Home Care Services", "moh", "policy", "Home care delivers nursing and personal care to patients in their homes through community care providers."),
    ("seed_telehealth", "Telemedicine Guidelines", "moh", "policy", "Telemedicine allows remote consultation when clinically appropriate and with proper documentation and consent."),
    ("seed_pharmacy_generic", "Generic Medicines", "moh", "policy", "Generic medicines contain the same active ingredients as originator drugs and are often more affordable."),
    ("seed_medicine_subsidy", "Medication Assistance", "moh", "subsidies", "Subsidy schemes help lower medication costs for eligible patients at public healthcare institutions."),
    ("seed_haze", "Haze Health Advisory", "hpb", "prevention", "During haze episodes, minimise outdoor activity and use N95 masks when PSI levels are high."),
    ("seed_dengue", "Dengue Prevention", "hpb", "prevention", "Dengue prevention includes removing stagnant water and using insect repellent in affected areas."),
    ("seed_hand_foot_mouth", "Hand Foot Mouth Disease", "hpb", "prevention", "HFMD is common in children; maintain hygiene and keep infected children home from school."),
    ("seed_chickenpox", "Chickenpox Vaccination", "healthhub", "prevention", "Varicella vaccination is recommended for children as part of the national immunisation programme."),
    ("seed_pap_smear", "Pap Smear Screening", "healthhub", "screening", "Pap smear screening detects cervical cell changes early in women aged 25 to 29 every three years."),
    ("seed_hpv_test", "HPV DNA Test", "healthhub", "screening", "HPV DNA testing is recommended for women aged 30 to 69 every five years for cervical cancer screening."),
    ("seed_fit_kit", "FIT Test Kit Usage", "healthhub", "screening", "The FIT kit collects stool samples at home for colorectal cancer screening; follow instructions carefully."),
    ("seed_mammogram_prep", "Mammogram Preparation", "healthhub", "screening", "Avoid deodorant on mammogram day and schedule after menstrual period for comfort."),
    ("seed_health_plan", "Personal Health Plan", "healthhub", "policy", "Healthier SG enrollees receive a personalised health plan from their enrolled family doctor."),
    ("seed_gp_first", "GP as First Touchpoint", "healthhub", "policy", "Primary care GPs serve as the first point of contact for non-emergency health concerns."),
    ("seed_a_and_e", "A&E Services", "healthhub", "policy", "Accident and Emergency departments handle life-threatening conditions; use GP or polyclinic for minor ailments."),
    ("seed_school_health", "School Health Services", "hpb", "prevention", "School health services provide vaccinations, dental checks, and health screening for students."),
    ("seed_workplace_health", "Workplace Health Promotion", "hpb", "prevention", "Workplace health programmes encourage physical activity, mental wellness, and smoke-free environments."),
    ("seed_sugar_intake", "Limiting Sugar Intake", "hpb", "prevention", "Excessive sugar intake contributes to obesity and diabetes; choose water over sugary drinks."),
    ("seed_sodium", "Reducing Sodium", "hpb", "prevention", "High sodium intake raises blood pressure; limit processed foods and added salt."),
    ("seed_sleep", "Healthy Sleep", "hpb", "prevention", "Adults should aim for 7 to 9 hours of quality sleep for physical and mental health."),
    ("seed_alcohol", "Alcohol Consumption", "hpb", "prevention", "Limit alcohol intake; excessive drinking increases liver disease and cancer risk."),
    ("seed_pdpa_notification", "PDPA Notification Obligation", "pdpc", "compliance", "Organisations must notify individuals of purposes for collection, use, and disclosure of personal data."),
    ("seed_pdpa_access", "PDPA Access and Correction", "pdpc", "compliance", "Individuals may request access to and correction of their personal data held by organisations."),
    ("seed_pdpa_protection", "PDPA Protection Obligation", "pdpc", "compliance", "Organisations must make reasonable security arrangements to protect personal data from unauthorised access."),
    ("seed_pdpa_retention", "PDPA Retention Limitation", "pdpc", "compliance", "Personal data should not be retained longer than necessary for legal or business purposes."),
    ("seed_pdpa_transfer", "PDPA Transfer Limitation", "pdpc", "compliance", "Cross-border data transfers require ensuring comparable protection standards in the recipient country."),
    ("seed_pdpa_dpo", "Data Protection Officer", "pdpc", "compliance", "Organisations must designate at least one DPO to oversee PDPA compliance and handle queries."),
    ("seed_pdpa_breach", "Data Breach Notification", "pdpc", "compliance", "Significant data breaches must be reported to PDPC and affected individuals without undue delay."),
    ("seed_pdpa_deemed", "Deemed Consent", "pdpc", "compliance", "Deemed consent may apply when individuals voluntarily provide data for a reasonable purpose."),
    ("seed_pdpa_anonymisation", "Anonymisation of Data", "pdpc", "compliance", "Anonymised data that cannot identify individuals may fall outside PDPA personal data scope."),
    ("seed_pdpa_research", "Research Use of Health Data", "pdpc", "compliance", "Research use of personal data requires consent or applicable exceptions with appropriate safeguards."),
    ("seed_pdpa_marketing", "PDPA and Direct Marketing", "pdpc", "compliance", "Organisations need consent or valid opt-out provisions before sending marketing messages."),
    ("seed_pdpa_employment", "Employee Personal Data", "pdpc", "compliance", "Employers must protect employee personal data and collect only what is necessary for employment purposes."),
    ("seed_emergency_treatment", "Emergency Treatment Consent", "pdpc", "compliance", "Emergency medical treatment may proceed without prior consent when patient is unable to consent."),
    ("seed_health_records", "Electronic Health Records", "moh", "policy", "NEHR enables authorised healthcare providers to access patient summary records for continuity of care."),
    ("seed_patient_portal", "HealthHub App", "healthhub", "policy", "HealthHub provides access to health records, appointment booking, medication refill, and health articles."),
    ("seed_medication_refill", "Medication Refill", "healthhub", "policy", "Patients can request medication refills through HealthHub for participating healthcare providers."),
    ("seed_appointment", "Medical Appointments", "healthhub", "policy", "Book polyclinic and selected specialist appointments through HealthHub or provider portals."),
    ("seed_bills_payment", "Medical Bills Payment", "healthhub", "policy", "View and pay hospital and polyclinic bills through HealthHub with linked payment methods."),
]

for slug, title, source, category, body in _TOPICS:
    EXTRA_SEED_DOCUMENTS.append(
        {
            "doc_id": slug,
            "source": source,
            "category": category,
            "title": title,
            "text": f"{title}\n\n{body} This information is based on publicly available Singapore health policy and health promotion guidance for educational purposes.",
        }
    )

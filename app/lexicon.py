"""Clinical lexicon: red flags, urgency tiers, negation cues, anatomy, glossary.

Curated for radiology + pathology report language. Every entry is a plain
string/regex used by the deterministic engine — no model weights required.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Urgency tiers
# ---------------------------------------------------------------------------
CRITICAL = "critical"   # seek emergency / immediate care
URGENT = "urgent"       # contact clinician within 24-48h
ROUTINE = "routine"     # follow up at next scheduled visit
BENIGN = "benign"       # reassuring / normal

TIER_ORDER = {CRITICAL: 3, URGENT: 2, ROUTINE: 1, BENIGN: 0}

TIER_LABELS = {
    CRITICAL: "Critical — seek immediate medical attention",
    URGENT: "Urgent — contact your doctor within 24–48 hours",
    ROUTINE: "Routine — discuss at your next appointment",
    BENIGN: "No urgent findings — normal / benign",
}

TIER_ADVICE = {
    CRITICAL: (
        "This report contains findings that can be time-sensitive. Contact your "
        "treating doctor or go to an emergency department now. Do not wait for a "
        "scheduled appointment."
    ),
    URGENT: (
        "These findings should be reviewed soon. Call your doctor's office within "
        "24–48 hours to discuss next steps. Seek emergency care sooner if symptoms worsen."
    ),
    ROUTINE: (
        "No time-sensitive findings detected. Bring this report to your next scheduled "
        "appointment and ask your doctor any questions you have."
    ),
    BENIGN: (
        "The report appears normal or describes benign findings. Still review it with "
        "your doctor, who knows your full history."
    ),
}

# ---------------------------------------------------------------------------
# Red-flag patterns -> (tier, human-readable reason)
# Patterns are matched case-insensitively against NEGATION-AWARE text spans.
# ---------------------------------------------------------------------------
RED_FLAGS: list[tuple[str, str, str]] = [
    # --- Critical: vascular / neuro / thoracic emergencies ---
    (r"pulmonary\s+embol(ism|us)|\bpe\b(?=.*(?:acute|occlus))", CRITICAL, "Possible pulmonary embolism (blood clot in lung)"),
    (r"acute\s+(?:ischemic\s+)?(?:stroke|infarct|cerebral\s+infarction)|large\s+vessel\s+occlusion", CRITICAL, "Signs of acute stroke / brain infarction"),
    (r"aortic\s+dissection|dissecting\s+aortic\s+aneurysm", CRITICAL, "Possible aortic dissection"),
    (r"ruptured\s+(?:abdominal\s+)?aortic\s+aneurysm|ruptured\s+aneurysm", CRITICAL, "Ruptured aneurysm"),
    (r"tension\s+pneumothorax|pneumothorax", CRITICAL, "Pneumothorax (collapsed lung)"),
    (r"epidural\s+hematoma|subdural\s+hematoma|subarachnoid\s+hemorrhage|intracranial\s+hemorrhage", CRITICAL, "Intracranial bleeding"),
    (r"midline\s+shift|mass\s+effect|uncal\s+herniation|herniation", CRITICAL, "Brain mass effect / herniation risk"),
    (r"free\s+air|pneumoperitoneum|perforation", CRITICAL, "Possible hollow-organ perforation (free air)"),
    (r"bowel\s+(?:ischemia|infarction|necrosis)|mesenteric\s+ischemia", CRITICAL, "Bowel ischemia / infarction"),
    (r"cardiac\s+tamponade|pericardial\s+tamponade", CRITICAL, "Cardiac tamponade"),
    (r"massive\s+(?:pleural\s+)?effusion", URGENT, "Large pleural effusion"),
    (r"acute\s+respiratory\s+(?:distress|failure)|ards", CRITICAL, "Acute respiratory distress"),
    (r"septic\s+shock|sepsis", CRITICAL, "Sepsis"),
    (r"testicular\s+torsion|ovarian\s+torsion", CRITICAL, "Organ torsion (blood supply cut off)"),
    (r"ectopic\s+pregnancy", CRITICAL, "Possible ectopic pregnancy"),
    (r"spinal\s+cord\s+compression|cauda\s+equina", CRITICAL, "Spinal cord / cauda equina compression"),
    (r"critical\s+(?:carotid|vertebral|renal|stenosis)|>\s*90\s*%\s*stenosis|9[0-9]\s*%\s*stenosis", CRITICAL, "Critical vessel stenosis"),
    (r"acute\s+cholecystitis|gangrenous\s+cholecystitis", URGENT, "Acute gallbladder inflammation"),
    (r"acute\s+appendicitis|perforated\s+appendix", URGENT, "Acute appendicitis"),
    (r"acute\s+pancreatitis|necrotizing\s+pancreatitis", URGENT, "Acute pancreatitis"),
    (r"acute\s+pyelonephritis", URGENT, "Kidney infection (pyelonephritis)"),
    (r"obstructing\s+(?:ureteral|renal)\s+(?:calculus|stone)|hydronephrosis", URGENT, "Obstructing kidney/ureter stone"),

    # --- Oncology / pathology flags ---
    (r"malignant|malignancy|carcinoma|adenocarcinoma|squamous\s+cell\s+carcinoma", URGENT, "Malignant (cancer) cells / diagnosis mentioned"),
    (r"metasta(?:sis|ses|tic)|metastatic\s+disease", URGENT, "Metastatic disease mentioned"),
    (r"invasive\s+(?:ductal|lobular)\s+carcinoma", URGENT, "Invasive carcinoma"),
    (r"high[\s-]?grade\s+(?:squamous\s+intraepithelial\s+lesion|dysplasia|neoplasia|lesion)|\bhsil\b", URGENT, "High-grade dysplasia / neoplasia"),
    (r"lymphoma|leukemia|melanoma|sarcoma|glioblastoma", URGENT, "Malignancy mentioned"),
    (r"suspicious\s+for\s+malignancy|cannot\s+exclude\s+malignancy|concerning\s+for\s+malignancy", URGENT, "Findings suspicious for malignancy"),
    (r"bi[\s-]?rads\s*(?:category\s*)?[45]|bi[\s-]?rads\s*[45]", URGENT, "BI-RADS 4/5 — suspicious breast lesion, biopsy advised"),
    (r"lung[\s-]?rads\s*[34]|lung[\s-]?rads\s*category\s*[34]", URGENT, "Lung-RADS 3/4 — nodule needs close follow-up"),
    (r"ti[\s-]?rads\s*[45]|tirads\s*[45]", URGENT, "TI-RADS 4/5 — suspicious thyroid nodule"),
    (r"pi[\s-]?rads\s*[45]|pirads\s*[45]", URGENT, "PI-RADS 4/5 — suspicious prostate lesion"),

    # --- Urgent: significant but not immediately life-threatening ---
    (r"deep\s+vein\s+thrombosis|\bdvt\b", URGENT, "Deep vein thrombosis (blood clot)"),
    (r"pulmonary\s+(?:edema|congestion)|congestive\s+heart\s+failure", URGENT, "Pulmonary edema / heart failure signs"),
    (r"pneumonia|consolidation", URGENT, "Pneumonia / consolidation"),
    (r"abscess", URGENT, "Abscess (collection of infection)"),
    (r"osteomyelitis", URGENT, "Bone infection (osteomyelitis)"),
    (r"fracture|dislocation", URGENT, "Fracture or dislocation"),
    (r"significant\s+(?:carotid|coronary|renal)\s+(?:artery\s+)?stenosis|moderate[\s-]to[\s-]severe\s+stenosis", URGENT, "Significant vessel narrowing"),
    (r"aneurysm", URGENT, "Aneurysm mentioned"),
    (r"struvite|staghorn\s+calculus", URGENT, "Large kidney stone"),
    (r"atypical\s+(?:cells|ductal\s+hyperplasia|lobular\s+hyperplasia)", URGENT, "Atypical cells — needs follow-up"),
    (r"positive\s+(?:for\s+)?(?:malignant\s+cells|cancer)", URGENT, "Positive for malignant cells"),
    (r"granulomatous\s+(?:inflammation|disease)|tuberculosis|\btb\b(?=.*(?:active|positive))", URGENT, "Granulomatous disease / possible TB"),
    (r"crohn'?s?\s+disease|ulcerative\s+colitis", ROUTINE, "Inflammatory bowel disease mentioned"),
    (r"cirrhosis", URGENT, "Cirrhosis of the liver"),
    (r"portal\s+hypertension|varices", URGENT, "Portal hypertension / varices"),

    # --- Routine follow-up findings ---
    (r"nodule|nodules", ROUTINE, "Nodule(s) noted — usually needs follow-up imaging"),
    (r"\bcysts?\b(?!\s+fluid)", ROUTINE, "Cyst(s) noted — commonly benign"),
    (r"mild\s+(?:stenosis|narrowing)", ROUTINE, "Mild narrowing noted"),
    (r"degenerative\s+(?:changes|disease)|spondylosis|osteoarthritis", ROUTINE, "Degenerative / arthritic changes"),
    (r"hepatic\s+steatosis|fatty\s+liver", ROUTINE, "Fatty liver noted"),
    (r"cholelithiasis|gallstones?", ROUTINE, "Gallstones noted"),
    (r"renal\s+(?:cyst|cysts)", ROUTINE, "Kidney cyst(s) noted"),
    (r"benign|no\s+evidence\s+of\s+malignancy|negative\s+for\s+malignancy", BENIGN, "Report states benign / no malignancy"),
    (r"normal\s+(?:study|examination|findings)|unremarkable|no\s+acute\s+(?:findings|disease|abnormality)", BENIGN, "Report reads as normal / unremarkable"),
]

# Compiled once.
RED_FLAGS_COMPILED = [(re.compile(p, re.IGNORECASE), tier, reason) for p, tier, reason in RED_FLAGS]

# ---------------------------------------------------------------------------
# Negation & uncertainty cues (ConNeg-style, tuned for radiology/pathology)
# ---------------------------------------------------------------------------
PRE_NEGATION = [
    r"no\s+(?:evidence\s+of\s+)?", r"without\s+", r"free\s+of\s+", r"negative\s+for\s+",
    r"not\s+(?:seen|identified|detected|demonstrated|appreciated)\s+", r"absence\s+of\s+",
    r"fails?\s+to\s+(?:show|demonstrate)\s+", r"rules?\s+out\s+", r"excludes?\s+",
    r"no\s+", r"none\s+", r"never\s+",
]
POST_NEGATION = [
    r"\s+(?:is|are|was|were)\s+(?:not|absent|negative)", r"\s+absent\b", r"\s+not\s+seen\b",
    r"\s+excluded\b", r"\s+ruled\s+out\b", r"\s+negative\b",
]
UNCERTAINTY = [
    r"cannot\s+exclude", r"cannot\s+rule\s+out", r"question\s+of", r"possible",
    r"possibly", r"suspicious\s+for", r"concerning\s+for", r"may\s+represent",
    r"could\s+represent", r"differential\s+includes", r"vs\.?", r"versus",
    r"probable", r"likely", r"suggestive\s+of", r"consistent\s+with",
    r"indeterminate", r"equivocal", r"borderline",
]

PRE_NEGATION_COMPILED = [re.compile(p, re.IGNORECASE) for p in PRE_NEGATION]
POST_NEGATION_COMPILED = [re.compile(p, re.IGNORECASE) for p in POST_NEGATION]
UNCERTAINTY_COMPILED = [re.compile(p, re.IGNORECASE) for p in UNCERTAINTY]

# ---------------------------------------------------------------------------
# Report section headers (radiology + pathology conventions)
# ---------------------------------------------------------------------------
SECTION_ALIASES: dict[str, list[str]] = {
    "clinical_history": ["clinical history", "history", "clinical information", "indication", "clinical indication", "reason for exam", "reason for study", "clinical"],
    "technique": ["technique", "method", "procedure", "exam", "examination type"],
    "comparison": ["comparison", "comparisons", "prior study", "prior studies", "previous study"],
    "findings": ["findings", "observations", "description", "gross description", "microscopic description", "microscopy", "microscopic", "gross"],
    "impression": ["impression", "conclusion", "conclusions", "summary", "diagnosis", "final diagnosis", "final impression", "interpretation", "result", "results", "synoptic report"],
    "recommendation": ["recommendation", "recommendations", "suggested", "advice", "follow-up", "follow up", "correlation"],
}

# ---------------------------------------------------------------------------
# Measurement extraction
# ---------------------------------------------------------------------------
MEASUREMENT_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(cm|mm|cm3|cc|ml|%|percent)\b", re.IGNORECASE
)
MEASUREMENT_CONTEXT_RE = re.compile(
    r"([A-Za-z][A-Za-z\s\-]{0,40}?)\s*(?:measur\w+|of|size|measuring)?\s*(?:up\s+to\s+|approximately\s+|about\s+)?(\d+(?:\.\d+)?)\s*(cm|mm|cm3|cc|ml|%|percent)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Patient-friendly glossary (term -> plain explanation)
# ---------------------------------------------------------------------------
GLOSSARY: dict[str, str] = {
    "consolidation": "An area of the lung filled with fluid or tissue instead of air, often seen with pneumonia.",
    "effusion": "Extra fluid collected in a body space (e.g., around the lung).",
    "pleural effusion": "Fluid collected in the space around the lung.",
    "nodule": "A small round spot seen on imaging. Most nodules are benign, but some need follow-up scans.",
    "mass": "An abnormal lump or growth larger than a nodule. Needs medical evaluation.",
    "lesion": "A general word for any abnormal area seen on a scan or under the microscope.",
    "stenosis": "Narrowing of a tube or vessel (e.g., an artery or the spinal canal).",
    "calcification": "Small calcium deposits. Often harmless and related to aging or old injury.",
    "edema": "Swelling caused by extra fluid in the tissues.",
    "pulmonary edema": "Fluid in the lungs, often related to heart function.",
    "infarct": "Tissue damage caused by loss of blood supply (e.g., a stroke in the brain).",
    "ischemia": "Reduced blood flow to a tissue or organ.",
    "thrombosis": "A blood clot inside a blood vessel.",
    "embolism": "A clot or material that traveled through the bloodstream and blocked a vessel.",
    "hemorrhage": "Bleeding.",
    "hematoma": "A collection of blood outside a blood vessel, like a deep bruise.",
    "pneumothorax": "Air leaked into the chest causing part of the lung to collapse.",
    "atelectasis": "Partial collapse or under-inflation of part of the lung.",
    "hepatomegaly": "An enlarged liver.",
    "splenomegaly": "An enlarged spleen.",
    "lymphadenopathy": "Enlarged lymph nodes, which can happen with infection or other conditions.",
    "steatosis": "Fat buildup in an organ, most often the liver ('fatty liver').",
    "cirrhosis": "Long-term scarring of the liver.",
    "hydronephrosis": "Swelling of a kidney because urine cannot drain properly.",
    "calculus": "A stone (e.g., kidney stone or gallstone).",
    "cholelithiasis": "Gallstones in the gallbladder.",
    "cholecystitis": "Inflammation of the gallbladder.",
    "pancreatitis": "Inflammation of the pancreas.",
    "appendicitis": "Inflammation of the appendix.",
    "diverticulosis": "Small pouches in the wall of the colon, common with age.",
    "diverticulitis": "Inflammation or infection of those colon pouches.",
    "osteophyte": "A small bone spur, usually from wear-and-tear arthritis.",
    "spondylosis": "Age-related wear of the spine (spinal arthritis).",
    "osteomyelitis": "Infection of a bone.",
    "abscess": "A pocket of infection (pus) in the body.",
    "biopsy": "A small tissue sample taken for lab examination.",
    "cytology": "Examination of individual cells under a microscope.",
    "histology": "Examination of tissue structure under a microscope.",
    "benign": "Not cancerous.",
    "malignant": "Cancerous.",
    "metastasis": "Cancer that has spread from where it started to another part of the body.",
    "carcinoma": "A cancer that starts in the lining (epithelial) cells of an organ.",
    "adenocarcinoma": "A cancer that starts in gland-like cells.",
    "dysplasia": "Abnormal-looking cells that are not yet cancer but may need monitoring.",
    "hyperplasia": "An increased number of cells, causing tissue enlargement.",
    "neoplasm": "A new abnormal growth; can be benign or malignant.",
    "in situ": "Abnormal cells that are still confined to where they started and have not spread.",
    "margin": "The edge of the tissue removed; 'clear margins' means no abnormal cells at the edge.",
    "grade": "How abnormal the cells look under the microscope; higher grade = more abnormal.",
    "stage": "How far a disease has spread in the body.",
    "immunohistochemistry": "Special stains used on tissue to identify cell types and guide diagnosis.",
    "bi-rads": "A standardized scoring system for breast imaging. 1–2 benign, 3 probably benign, 4–5 suspicious.",
    "lung-rads": "A standardized scoring system for lung nodules on CT. Higher scores need closer follow-up.",
    "ti-rads": "A standardized scoring system for thyroid nodules on ultrasound.",
    "pi-rads": "A standardized scoring system for prostate lesions on MRI.",
    "ejection fraction": "The percentage of blood the heart pumps out with each beat. Normal is about 55–70%.",
    "unremarkable": "Radiologist shorthand for 'normal — nothing to worry about here'.",
    "correlate clinically": "The radiologist recommends the doctor match these findings with your symptoms and labs.",
    "interval change": "Difference compared with your previous scan.",
    "stable": "No change compared with prior imaging — usually reassuring.",
    "interval development": "Something new since your last scan.",
    "ground-glass opacity": "A hazy area on lung CT, often from inflammation or infection.",
    "opacity": "An area on an X-ray or CT that appears whiter than usual.",
    "echogenicity": "How bright tissue appears on ultrasound.",
    "hypoechoic": "Darker than surrounding tissue on ultrasound.",
    "hyperechoic": "Brighter than surrounding tissue on ultrasound.",
    "attenuation": "How much tissue blocks X-rays on CT; described as low or high.",
    "enhancement": "How much a tissue brightens after contrast dye is given.",
    "contrast": "A dye given by mouth or IV to make structures show up better on imaging.",
    "mri": "Magnetic Resonance Imaging — detailed pictures using magnets, no radiation.",
    "ct": "Computed Tomography — cross-section X-ray pictures.",
    "ultrasound": "Imaging using sound waves, no radiation.",
    "pet-ct": "A scan combining metabolic activity (PET) with anatomy (CT), often used in cancer.",
    "mammogram": "X-ray imaging of the breast.",
    "pathologist": "The doctor who examines tissue samples under the microscope.",
    "radiologist": "The doctor who interprets imaging scans.",
    "occult": "Hidden; not visible to the eye.",
    "patent": "Open and unblocked (e.g., a blood vessel).",
    "occluded": "Blocked.",
    "dilated": "Wider than normal.",
    "tortuous": "Twisted or winding (often used for blood vessels).",
    "sclerosis": "Hardening of tissue.",
    "fibrosis": "Scar-like tissue.",
    "atrophy": "Shrinkage or thinning of tissue.",
    "hypertrophy": "Thickening or enlargement of tissue.",
    "polyp": "A small growth on a lining surface, e.g., in the colon or gallbladder.",
    "stricture": "An abnormal narrowing of a tube or passage.",
    "fistula": "An abnormal connection between two body spaces.",
    "thrombus": "A blood clot.",
    "embolus": "A clot or material traveling through the bloodstream.",
    "infarction": "Tissue death from loss of blood supply.",
    "necrosis": "Tissue death.",
    "gangrenous": "Tissue death due to loss of blood supply, often with infection.",
    "sepsis": "A severe body-wide response to infection.",
    "septic": "Related to sepsis or infection.",
    "febrile": "Having a fever.",
    "afebrile": "Without fever.",
    "hemodynamically stable": "Blood pressure and heart rate are steady.",
    "prognosis": "The expected course and outcome of a disease.",
    "differential diagnosis": "The list of possible conditions that could explain the findings.",
    "idiopathic": "Of unknown cause.",
    "congenital": "Present from birth.",
    "acute": "Sudden onset, recent.",
    "chronic": "Long-standing, developing over time.",
    "bilateral": "On both sides.",
    "unilateral": "On one side.",
    "proximal": "Closer to the center of the body.",
    "distal": "Farther from the center of the body.",
    "anterior": "Toward the front.",
    "posterior": "Toward the back.",
    "superior": "Upper.",
    "inferior": "Lower.",
    "medial": "Toward the middle.",
    "lateral": "Toward the side.",
    "apical": "At the top (apex), e.g., of the lung.",
    "basal": "At the base/bottom.",
    "lobar": "Relating to a lobe of an organ (e.g., lung lobe).",
    "subsegmental": "In a small branch (e.g., of a lung airway or artery).",
    "hilar": "At the hilum — the central area where vessels enter an organ.",
    "mediastinal": "In the mediastinum — the central chest space between the lungs.",
    "pericardial": "Around the heart.",
    "peritoneal": "Relating to the abdominal lining.",
    "retroperitoneal": "Behind the abdominal lining.",
    "intracranial": "Inside the skull.",
    "extracranial": "Outside the skull.",
    "cervical": "Relating to the neck (or cervix, depending on context).",
    "thoracic": "Relating to the chest.",
    "lumbar": "Relating to the lower back.",
    "sacral": "Relating to the sacrum (base of spine).",
    "femoral": "Relating to the thigh / femur.",
    "popliteal": "Behind the knee.",
    "iliac": "Relating to the pelvis bones / iliac vessels.",
    "renal": "Relating to the kidney.",
    "hepatic": "Relating to the liver.",
    "splenic": "Relating to the spleen.",
    "pancreatic": "Relating to the pancreas.",
    "adrenal": "Relating to the adrenal glands above the kidneys.",
    "thyroid": "A butterfly-shaped gland in the neck that controls metabolism.",
    "prostate": "A gland in males below the bladder.",
    "ovarian": "Relating to the ovaries.",
    "uterine": "Relating to the uterus.",
    "endometrial": "Relating to the lining of the uterus.",
    "mammographic": "Relating to mammography (breast X-ray).",
    "sonographic": "Relating to ultrasound.",
    "fluoroscopic": "Relating to real-time X-ray imaging.",
    "mammotome": "A device used for breast biopsy.",
    "fnac": "Fine Needle Aspiration Cytology — cells drawn out with a thin needle.",
    "core biopsy": "A biopsy using a larger needle to take a tissue cylinder.",
    "excisional biopsy": "Removing the whole lesion for examination.",
    "incisional biopsy": "Removing part of a lesion for examination.",
    "frozen section": "Rapid tissue examination during surgery.",
    "tumor marker": "A substance (often in blood) that can indicate certain cancers.",
    "cea": "Carcinoembryonic antigen — a tumor marker blood test.",
    "afp": "Alpha-fetoprotein — a tumor marker blood test.",
    "psa": "Prostate-specific antigen — a blood test for the prostate.",
    "ca-125": "A tumor marker often used for ovarian cancer.",
    "ca 19-9": "A tumor marker often used for pancreatic/biliary cancers.",
    "her2": "A protein tested in some cancers (e.g., breast) to guide treatment.",
    "er/pr": "Estrogen/progesterone receptors — tested in breast cancer to guide treatment.",
    "ki-67": "A marker of how fast cells are dividing.",
    "pd-l1": "A marker used to predict response to immunotherapy.",
    "microsatellite": "Short repeated DNA sequences; instability can guide cancer treatment.",
    "next-generation sequencing": "Broad DNA testing of a tumor to find targetable mutations.",
}

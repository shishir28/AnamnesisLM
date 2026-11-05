"""Sentence templates for rendering facts into notes.

Each key maps to several phrasings. record_builder picks one per sentence, so a note mixes
styles and the model has to learn what a phrase means, not one fixed pattern per label.

HOLDOUT_TEMPLATES is never used for training. Its phrasings and multi-word fragments must stay
out of TRAIN_TEMPLATES, but its single words must all come from TRAIN_TEMPLATES;
tests/data/test_templates.py checks both.
"""

Templates = dict[str, list[str]]

TRAIN_TEMPLATES: Templates = {
    # condition, assertion affirmed, temporality current
    "active_condition": [
        "Currently has {mention}.",
        "Active problem: {mention}.",
        "The patient is being treated for {mention}.",
        "Dx: {mention}, ongoing.",
        "Assessment includes {mention}, which remains active.",
        "Ongoing {mention}.",
        "{mention} is an active diagnosis.",
        "Being managed for {mention}.",
        "Diagnosed with {mention}, still active.",
        "Problem: {mention}, active.",
        "{mention}, currently under treatment.",
        "Has a current diagnosis of {mention}.",
    ],
    # condition, assertion affirmed, temporality historical
    "resolved_condition": [
        "History of {mention}.",
        "Past problem: {mention}.",
        "The patient previously had {mention}, now resolved.",
        "PMH: {mention}, resolved.",
        "Prior {mention} has since resolved.",
        "Had {mention} in the past.",
        "{mention}, resolved.",
        "Recovered from {mention}.",
        "Previous episode of {mention}, resolved.",
        "Former diagnosis: {mention}.",
        "{mention}, treated and resolved.",
        "Was treated for {mention} previously.",
    ],
    # medication, affirmed, current
    "medication": [
        "Takes {mention}.",
        "Medication: {mention}.",
        "The patient is taking {mention}.",
        "Meds: {mention}.",
        "Continues on {mention}.",
        "On {mention}.",
        "Currently taking {mention}.",
        "Prescribed {mention}.",
        "Rx: {mention}.",
        "Remains on {mention}.",
        "Active medication: {mention}.",
        "Uses {mention} regularly.",
    ],
    # not rendered yet (frequency is never filled); kept for the dose/frequency decision
    "medication_with_frequency": [
        "Takes {mention} {frequency}.",
        "Medication: {mention}, {frequency}.",
        "The patient is taking {mention} {frequency}.",
        "Meds: {mention} {frequency}.",
        "Continues on {mention} {frequency}.",
    ],
    # allergy, affirmed, current
    "allergy": [
        "Allergic to {mention}.",
        "Allergy: {mention}.",
        "The patient reports an allergy to {mention}.",
        "ALL: {mention}.",
        "Documented allergy: {mention}.",
        "Allergies: {mention}.",
        "Has an allergy to {mention}.",
        "{mention} allergy.",
        "Reacts to {mention}.",
        "Allergy to {mention} noted.",
        "Sensitive to {mention}.",
        "Reports an allergic reaction to {mention}.",
    ],
    # symptom (kind condition), current
    "symptom_affirmed": [
        "Positive for {mention}.",
        "The patient reports {mention}.",
        "{mention} present.",
        "Evidence of {mention}.",
        "Complains of {mention}.",
        "Endorses {mention}.",
        "Reports {mention}.",
        "Experiencing {mention}.",
        "c/o {mention}.",
        "Has {mention} today.",
    ],
    "symptom_negated": [
        "Negative for {mention}.",
        "The patient denies any {mention}.",
        "No {mention}.",
        "No evidence of {mention}.",
        "Denies {mention}.",
        "No complaints of {mention}.",
        "Without {mention}.",
        "{mention} absent.",
        "Free of {mention}.",
        "No signs of {mention}.",
        "{mention} denied.",
    ],
    "symptom_uncertain": [
        "Possible {mention}.",
        "The patient may have {mention}.",
        "Suspected {mention}.",
        "Query {mention}.",
        "Probable {mention}.",
        "{mention} is suspected.",
        "Questionable {mention}.",
        "{mention}, uncertain.",
        "Possibly {mention}.",
        "{mention}, unconfirmed.",
    ],
    # condition, affirmed, current, experiencer family
    "family": [
        "{family_member} has {mention}.",
        "Family history: {mention} ({family_member}).",
        "{family_member} has a history of {mention}.",
        "FHx: {family_member} w/ {mention}.",
        "{family_member} was diagnosed with {mention}.",
        "{family_member} had {mention}.",
        "{mention} runs in the family ({family_member}).",
        "{family_member} with {mention}.",
        "{family_member} is treated for {mention}.",
        "Family history positive for {mention} ({family_member}).",
        "{family_member} suffers from {mention}.",
    ],
}

# Generalisation test set only: new phrasings built from words the training templates already
# use. It measures whether the model understands known words in a new order. Unseen *words*
# (e.g. "prescription") cannot be learned without pretraining, and a single such phrasing per
# key made scores swing 35-71% between seeds, so holdout wording stays inside the training
# vocabulary and each key has several phrasings to average over.
HOLDOUT_TEMPLATES: Templates = {
    "active_condition": [
        "Currently being treated for {mention}.",
        "Ongoing problem: {mention}.",
        "{mention} remains an active problem.",
        "Under treatment for {mention}, ongoing.",
    ],
    "resolved_condition": [
        "Previously treated for {mention}, resolved.",
        "Resolved: {mention}.",
        "Past episode of {mention}.",
        "{mention} was treated in the past.",
    ],
    "medication": [
        "Currently on {mention}.",
        "Takes {mention} regularly.",
        "Current medication: {mention}.",
        "The patient remains on {mention}.",
    ],
    "medication_with_frequency": [
        "Currently on {mention}, {frequency}.",
        "Current medication: {mention} {frequency}.",
    ],
    "allergy": [
        "Documented allergy to {mention}.",
        "Allergies noted: {mention}.",
        "The patient is allergic to {mention}.",
        "Has a documented allergy to {mention}.",
    ],
    "symptom_affirmed": [
        "Reports {mention} today.",
        "The patient complains of {mention}.",
        "Positive evidence of {mention}.",
        "{mention} present today.",
    ],
    "symptom_negated": [
        "The patient reports no {mention}.",
        "Denies {mention} today.",
        "{mention} is absent.",
        "Without any {mention}.",
    ],
    "symptom_uncertain": [
        "Possible {mention}, unconfirmed.",
        "{mention} is possible.",
        "Suspected {mention}, uncertain.",
        "{mention} is probable.",
    ],
    "family": [
        "Family history of {mention} ({family_member}).",
        "{family_member} is diagnosed with {mention}.",
        "FHx positive for {mention} ({family_member}).",
        "{family_member} has had {mention}.",
    ],
}

# Augmentation vocabularies. Large enough that no single name becomes a cue for a label
# (with only three family conditions, "Asthma" alone started to mean "family").
# Symptoms must not contain cue words such as "no" (see tests/data/test_templates.py).
SYMPTOMS = [
    "chest pain", "fever", "cough", "shortness of breath", "headache", "nausea",
    "vomiting", "dizziness", "fatigue", "abdominal pain", "back pain", "sore throat",
    "runny nose", "joint pain", "rash", "diarrhea", "constipation", "palpitations",
    "chills", "wheezing", "blurred vision", "insomnia", "weight loss", "ankle swelling",
    "muscle aches",
]  # fmt: skip
FAMILY_MEMBERS = [
    "Mother", "Father", "Sister", "Brother", "Daughter", "Son",
    "Maternal grandmother", "Paternal grandfather", "Aunt", "Uncle",
]  # fmt: skip
# Includes conditions patients also have, so the sentence, not the name, decides experiencer.
FAMILY_CONDITIONS = [
    "Hypertension", "Diabetes mellitus type 2", "Asthma", "Hyperlipidemia",
    "Coronary heart disease", "Breast cancer", "Colon cancer", "Stroke",
    "Alzheimer's disease", "Osteoporosis", "Chronic kidney disease", "Atrial fibrillation",
    "Depression", "Rheumatoid arthritis", "Prostate cancer", "Hypothyroidism",
    "Glaucoma", "Migraine", "Obesity", "Epilepsy", "Anemia", "Gout",
    "Chronic obstructive lung disease", "Parkinson's disease", "Lung cancer",
]  # fmt: skip

TEMPLATE_FAMILIES: list[dict[str, str]] = [
    {
        "name": "plain",
        "active_condition": "Currently has {mention}.",
        "resolved_condition": "History of {mention}.",
        "medication": "Takes {mention}.",
        "medication_with_frequency": "Takes {mention} {frequency}.",
        "allergy": "Allergic to {mention}.",
        "symptom_affirmed": "Positive for {mention}.",
        "symptom_negated": "Negative for {mention}.",
        "symptom_uncertain": "Possible {mention}.",
        "family": "{family_member} has {mention}.",
    },
    {
        "name": "problem_list",
        "active_condition": "Active problem: {mention}.",
        "resolved_condition": "Past problem: {mention}.",
        "medication": "Medication: {mention}.",
        "medication_with_frequency": "Medication: {mention}, {frequency}.",
        "allergy": "Allergy: {mention}.",
        "symptom_affirmed": "Positive for {mention}.",
        "symptom_negated": "Negative for {mention}.",
        "symptom_uncertain": "Possible {mention}.",
        "family": "Family history: {mention} ({family_member}).",
    },
    {
        "name": "narrative",
        "active_condition": "The patient is being treated for {mention}.",
        "resolved_condition": "The patient previously had {mention}, now resolved.",
        "medication": "The patient is taking {mention}.",
        "medication_with_frequency": "The patient is taking {mention} {frequency}.",
        "allergy": "The patient reports an allergy to {mention}.",
        "symptom_affirmed": "The patient reports {mention}.",
        "symptom_negated": "The patient denies any {mention}.",
        "symptom_uncertain": "The patient may have {mention}.",
        "family": "{family_member} has a history of {mention}.",
    },
    {
        "name": "clinician_shorthand",
        "active_condition": "Dx: {mention}, ongoing.",
        "resolved_condition": "PMH: {mention}, resolved.",
        "medication": "Meds: {mention}.",
        "medication_with_frequency": "Meds: {mention} {frequency}.",
        "allergy": "NKDA except {mention}.",
        "symptom_affirmed": "{mention} present.",
        "symptom_negated": "No {mention}.",
        "symptom_uncertain": "Possible {mention}.",
        "family": "FHx: {family_member} w/ {mention}.",
    },
    {
        "name": "assessment",
        "active_condition": "Assessment includes {mention}, which remains active.",
        "resolved_condition": "Prior {mention} has since resolved.",
        "medication": "Continues on {mention}.",
        "medication_with_frequency": "Continues on {mention} {frequency}.",
        "allergy": "Documented allergy: {mention}.",
        "symptom_affirmed": "Evidence of {mention}.",
        "symptom_negated": "No evidence of {mention}.",
        "symptom_uncertain": "Possible {mention}.",
        "family": "{family_member} was diagnosed with {mention}.",
    },
    {
        "name": "history_section",
        "active_condition": "Present illness: {mention}.",
        "resolved_condition": "Past medical history is notable for {mention}, no longer active.",
        "medication": "Current prescription: {mention}.",
        "medication_with_frequency": "Current prescription: {mention}, taken {frequency}.",
        "allergy": "Known allergy to {mention}.",
        "symptom_affirmed": "The patient reports {mention}.",
        "symptom_negated": "The patient reports no {mention}.",
        "symptom_uncertain": "The patient may have {mention}.",
        "family": "{family_member}'s history includes {mention}.",
    },
]

SYMPTOMS = ["chest pain", "fever", "cough", "shortness of breath", "headache", "nausea"]
FAMILY_MEMBERS = ["Mother", "Father", "Sister", "Brother"]
FAMILY_CONDITIONS = ["Hypertension", "Diabetes mellitus type 2", "Asthma"]

# Rendered only for the generalisation test set, never for training.
HOLDOUT_FAMILY = "history_section"
TRAIN_FAMILIES = [f for f in TEMPLATE_FAMILIES if f["name"] != HOLDOUT_FAMILY]

# AnamnesisLM — Development Plan and Scope of Work

Status: proposed learning project. AnamnesisLM is a small decoder-only language model for clinical fact extraction; the name comes from *anamnesis*, the clinical term for a patient's history. This plan merges the earlier clinical fact extraction plan with the PhysioNet access steps, compute sizing, and later pretraining and EHR-task extensions from the earlier ClinicalSparkLM plan; both source plans have since been removed. It builds on the tokenizer, attention, next-token prediction, training, and evaluation work in SparkLM.

See [ANAMNESISLM_PLAN.mmd](ANAMNESISLM_PLAN.mmd) for the development blocks and flow.

**Multi-day means your development effort:** reading, designing, manually writing code, debugging, and interpreting experiments over several days or longer. It does not mean a commitment to multi-day training. Training duration is estimated separately from measurements on your DGX Spark.

You implement the project yourself. Agent assistance should provide explanations, reviews, file outlines, and debugging support when requested. This document defines future work; it does not mark existing components as validated or create implementation files.

This is a separate applied-project roadmap. [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) remains the source of truth for the book curriculum. Reuse the relevant book components after their correctness and behavior are understood.

## Objective and first release

Build a small decoder-only language model with randomly initialized weights that converts a short clinical note into structured facts with supporting text. The first release is a local research and learning prototype.

Example input: “History of asthma. Denies chest pain. Mother has diabetes.”

Expected distinctions: asthma belongs to the patient and is historical; chest pain is explicitly negated; diabetes belongs to a family member. Every extracted fact must point to the phrase that supports it.

Completing the first release demonstrates a specialized extractor, not a general clinical base model. A biomedical or clinical base model is additional work in the later extensions.

The first release includes conditions and symptoms, medications, allergies, assertion status, patient/family attribution, and evidence spans. Develop these in increments, beginning with conditions and symptoms. Medication dose, unit, route, and frequency are extracted only when explicitly stated.

## Scope of work

### Included in the first release

- Apply for PhysioNet access early so approval arrives before the later extensions need it.
- Inspect and document a synthetic patient dataset; build reproducible note/target examples.
- Define and version a JSON output schema and annotation rules.
- Implement deterministic patient splits, template holdouts, tokenization, batching, target alignment, and loss masks.
- Reuse and inspect your manual Transformer; train its weights from random initialization.
- Implement task training, evaluation, checkpoint save/load, and generation explicitly.
- Build a simple rule-based extraction baseline and compare it with the model.
- Create a local CLI for entering a note and displaying JSON, supporting phrases, and validation errors.
- Record configuration, dataset version, seed, code version, metrics, performance, and conclusions.

### Later extensions

Separate milestones after the extraction prototype; see [Later extensions](#later-extensions-1):

- Domain next-token pretraining on PubMed and PMC-OA, and a domain tokenizer.
- Continued pretraining and evaluation on credentialed real notes (MIMIC-IV-Note).
- ICD-10 coding from discharge summaries and radiology impression generation.
- Patient timeline summarization, FHIR mapping, terminology normalization, larger models.

### Out of scope

- Models of 1B parameters or more (weeks of training on one Spark).
- Clinical diagnosis, treatment recommendations, or any patient-facing use.
- Production EHR integration.
- A general medical chatbot.
- Sending credentialed data (MIMIC, n2c2) or derived artifacts to any online service, including coding agents.

## Output contract

Use a top-level `facts` list. Each fact has the following fields; publish the exact JSON schema before generating training examples.

| Field | Meaning |
|---|---|
| `kind` | `condition`, `symptom`, `medication`, or `allergy` |
| `mention` | The entity phrase as written in the note; no inferred standardized code |
| `assertion` | `affirmed`, `negated`, or `uncertain` |
| `experiencer` | `patient`, `family`, or `unspecified` |
| `temporality` | `current`, `historical`, or `unspecified` |
| `evidence` | Supporting text copied from the original note |
| `evidence_start`, `evidence_end` | Zero-based Python character offsets, with an exclusive end |
| `dose`, `unit`, `route`, `frequency` | Medication fields; `null` when unstated or inapplicable |

Require `note[evidence_start:evidence_end] == evidence`. Evidence should include the contextual cue needed to justify assertion or attribution. Define stable fact ordering for target serialization and one-to-one matching for evaluation.

An unmentioned fact is unknown, not a negative finding. An explicit “no known allergies” statement needs a documented representation distinct from an empty allergy list; settle that rule in the schema milestone. A medication mention does not establish that it is currently taken or prescribed. Do not invent missing values.

## Learning and development milestones

These are suggested blocks of active work, not deadlines or training-time estimates. Several blocks may take more than one day.

| Milestone | Suggested development effort | What you write or investigate | Completion evidence |
|---|---|---|---|
| 0. Setup | 1 session | PhysioNet application, data manifest template, external directory for credentialed data with a Claude Code `deny` rule | Application submitted; manifest template exists; dummy-file test shows agent file tools and shell commands cannot read the credentialed-data path |
| 1. Task and schema | 1 session | Schema, annotation rules, and 20 hand-written examples covering negation, history, family, uncertainty, and absent information | Every example has an unambiguous expected output and valid evidence offsets |
| 2. Data inspection | 1 session | Synthea reader and provenance record; inspect 100–1,000 patients initially | Record formats, available fields, counts, version/hash, and limitations documented |
| 3. Example construction | 1–2 sessions | Narrative rendering, explicit context variations, evidence offsets, duplicate detection, and patient/template splits | Repeated preparation with the same inputs and seed produces identical examples and splits |
| 4. Batching and loss | 1–2 sessions | Prompt serialization, response boundaries, EOS/padding, batching, and response-only loss | Inspect token IDs, masks, and one-position target shifts on a tiny example |
| 5. Model and tiny overfit | 1–2 sessions | Exact parameter count, configuration, training loop, and checkpoint round trip | Tiny deterministic training set can be overfit; reloaded logits match |
| 6. Baseline and first run | 1–2 sessions | Rule extractor, a short device benchmark, and the first recorded task-training run | Baseline metrics, model metrics, tokens/second, peak memory, and checkpoint recorded |
| 7. Error analysis and iteration | 1–2 sessions | Categorize validation errors and change one important variable at a time | Explain whether an intervention improves held-out behavior and what still fails |
| 8. Local demonstration | 1 session | CLI, schema/evidence validation, and a small fixed demonstration suite | A saved model runs on new notes and exposes invalid or unsupported output |
| 9. Final evaluation | 1 session | Freeze settings, evaluate once on the reserved test set, and write the report | Reproducible result, baseline comparison, limitations, and next decision |

The initial planning allowance is roughly 11–15 active sessions, spread across the days that suit you. This includes learning and debugging; it is not a guarantee of model accuracy or project duration. PhysioNet approval (days to a few weeks) runs in parallel with milestones 1–9, so nothing waits on it.

## Milestone 0: setup details

### Apply for PhysioNet access

PhysioNet (physionet.org, run by MIT) hosts MIMIC and other de-identified hospital data. Access requires approval.

1. Create an account at physionet.org.
2. Complete the CITI Program course "Data or Specimens Only Research" (free, a few hours). Register under "Massachusetts Institute of Technology Affiliates". Download the completion report.
3. Request credentialed access on the PhysioNet profile: upload the CITI report, give a short research description, and name a reference (manager or colleague).
4. After approval, sign the Data Use Agreement (DUA) on each dataset page (for example MIMIC-IV-Note), then download.

n2c2/i2b2 datasets (Harvard DBMI portal) have a separate application and DUA.

### Record a data manifest

One entry per dataset: source, licence or DUA, version, download date, privacy decision, and preprocessing steps. See [DATA_POLICY.md](../essentials/DATA_POLICY.md).

### Keep credentialed data outside the repo

Store MIMIC and n2c2 data outside SparkLM, for example `/data/physionet/`, and add a deny rule to `.claude/settings.json`. An absolute path needs a leading `//`; a single `/` is resolved relative to the settings file ([path pattern rules](https://code.claude.com/docs/en/permissions#how-path-patterns-match)):

```json
{ "permissions": { "deny": ["Read(//data/physionet/**)"] } }
```

A `Read` rule covers Claude Code's file tools, not arbitrary shell subprocesses, so it is not proof of isolation on its own. Before downloading credentialed data, place harmless dummy files at the path and confirm that the agent's file tools and shell commands (for example `cat`, `head`, Python) all fail to read them. Apply equivalent controls to any other coding agent used in this repository.

The PhysioNet DUA forbids sending MIMIC text to online LLM services, and coding agents send file contents they read to their API. Training locally on the Spark is allowed. MIMIC maintainers also advise treating derived datasets and models as sensitive and sharing them only through PhysioNet under the source agreement.

## Data construction and split rules

Start with Synthea records and your own deterministic narrative templates. Structured records supply positive facts, but generally do not supply all the narrative cues needed for negation, uncertainty, and family attribution. Create those examples deliberately and label them from what the generated note actually says. Do not infer negative findings from missing record fields.

Begin with a small inspectable sample. Expand the number of patients and examples only after the renderer, labels, and evidence spans are correct. Record unique patients, unique examples, unique tokens, and total training tokens processed separately; repeated epochs do not create new data diversity.

- Assign patients to train/validation/test before producing examples; an initial 80/10/10 allocation is a planning choice to confirm during preparation.
- Reserve separate template families for a generalization evaluation. Keep ordinary unseen-patient evaluation and unseen-template evaluation distinguishable.
- Remove duplicate and near-duplicate notes across splits. Deduplicate reused raw text before any optional language-model pretraining.
- Keep the 20 design examples separate from the blind test set because they guide development.
- Do not use test patients, their records, or their narratives in optional pretraining.
- Maintain a small independently written, held-out challenge set so success cannot be explained solely by copying renderer patterns.

Synthetic results establish performance on the specified synthetic task. Real-note claims require a separately annotated evaluation set and domain review; MIMIC notes are not ready-made labels for this extraction schema.

## Model and training approach

1. Start with a tiny model for data and loss correctness. Then choose an approximately 30–60M parameter configuration, counting embeddings and the vocabulary projection explicitly. Consider about 125M only after smaller-model results justify it.
2. Reuse the current tokenizer initially. Keeping a tokenizer vocabulary does not load pretrained model weights. A clinical tokenizer is a later experiment with vocabulary learned only from training data; preserve the character reference path.
3. Start with a 512-token total sequence budget for prompt, note, response, and EOS. Count the full serialized example. Reject and count examples that exceed the budget during the pilot; do not silently cut off facts, supporting evidence, or targets. Revisit the budget if representative examples do not fit.
4. Train the randomly initialized model directly on paired extraction examples first. This is a valid supervised training baseline and avoids making an unlabeled pretraining corpus a dependency for the first working version.
5. During task training, supervise response tokens and the response EOS. Ignore prompt and padding targets. Verify the mask after next-token shifting; the first response token must remain supervised.
6. Use deterministic greedy generation for evaluation. Initially process one note at a time, stop at EOS or the generation limit, and report incomplete or invalid responses.
7. Explore domain next-token pretraining as a separate experiment after the baseline (Extension A). Use compatible, documented training text and compare the resulting extractor on identical held-out data. Record additional token and compute budgets so any benefit has context.

The initial task is structured extraction, not open-ended question answering. A small randomly initialized model may learn the synthetic task while generalizing poorly to real clinical language; that is a result to measure rather than hide by increasing model size.

## Runtime, checkpoints, and experiment records

Development effort is scheduled above. Estimate training time independently: warm up the device, measure representative optimizer steps including the intended sequence length and batch size, and record sustained tokens/second and peak memory. Estimate wall time from the planned tokens plus evaluation and checkpoint overhead. Re-benchmark after changing model size or sequence length.

For each meaningful experiment record the question, hypothesis, code commit or dirty-worktree note, schema/data/tokenizer versions, split IDs, seed, resolved model/training settings, exact parameter count, device/software, precision, tokens processed, losses, task metrics, runtime, memory, and conclusion. Use the conventions in [IMPLEMENTATION_PLAN.md §12](../IMPLEMENTATION_PLAN.md#12-experiment-record).

Checkpoints include model and optimizer state, resolved configuration, tokenizer/schema identity, training position, dataset identity, and relevant RNG state. If exact resumption is claimed, preserve sampler/data-order state too and compare a resumed run with uninterrupted execution. Begin with a fixed-batch, dropout-disabled check, then validate the actual shuffled/dropout-enabled path before claiming full reproducibility.

Keep raw and prepared data under ignored `data/raw/`, `data/interim/`, and `data/processed/`; weights under ignored `checkpoints/`; and generated run artifacts under ignored `experiments/runs/`. Commit safe manifests, concise summaries, and learning notes. Keep credentialed data outside the repo as described in milestone 0.

## Evaluation and acceptance criteria

Correctness checks precede scale: tokenizer round trips, input/target alignment, causal isolation, response/padding loss masks, tiny overfit, evidence offsets, patient separation, and checkpoint reload. Add focused tests as you implement each invariant.

Compare the model and rule baseline on identical examples using:

- JSON parse rate and schema validity; invalid outputs remain failures in overall scores.
- Entity precision, recall, and F1 by fact kind, with a documented one-to-one matching rule.
- Strict fact F1 requiring the entity and its assertion, experiencer, and temporality to match.
- Negation, uncertainty, and family-attribution accuracy on the relevant labeled subsets.
- Medication field accuracy with denominators that distinguish explicitly supplied values from missing values.
- Evidence substring/offset validity and agreement with annotated supporting spans. A matching substring alone does not prove that an extracted fact is supported.
- Unsupported-fact rate, omissions, output latency, memory, parameter count, and checkpoint size.

Provisional learning targets are at least 95% schema-valid output and at least 0.85 strict fact micro-F1 on held-out synthetic notes. These are proposed project goals, not expected or demonstrated performance. Agree and freeze the scoring rules and goals before the main run, report each fact kind separately, and retain failed runs. Judge utility against the rule baseline; do not require or imply that the neural model must beat rules on template-generated text.

The first release is complete when you can reproduce the data preparation and training commands, reload a checkpoint, run the CLI, produce the final reserved-test report, explain the main errors, and trace one example through tokens, logits, masked loss, generation, and validated facts. Distinguish engineering completion from reaching the quality targets.

## Planned files and deliverables

These paths are a proposal for when each milestone starts. Create small empty files as needed and write their implementations yourself; reuse existing validated SparkLM modules rather than duplicating the Transformer.

| Area | Planned files |
|---|---|
| Task definition | `docs/ANAMNESISLM_SCHEMA.md`, `docs/ANAMNESISLM_ANNOTATION_GUIDE.md` |
| Data and labels | `src/sparklm/ehr/data.py`, `src/sparklm/ehr/narratives.py`, `src/sparklm/ehr/schema.py` |
| Task batches and loss | `src/sparklm/ehr/batching.py`, `src/sparklm/ehr/training.py` |
| Baseline and evaluation | `src/sparklm/ehr/baseline.py`, `src/sparklm/ehr/metrics.py`, `src/sparklm/ehr/inference.py` |
| Entry points | `scripts/prepare_ehr.py`, `scripts/train_ehr.py`, `scripts/evaluate_ehr.py`, `scripts/ehr_cli.py` |
| Focused checks | `tests/test_ehr_data.py`, `tests/test_ehr_schema.py`, `tests/test_ehr_batching.py`, `tests/test_ehr_metrics.py`, `tests/test_ehr_inference.py` |
| Records | `experiments/ehr/` for safe configurations and run summaries; `LEARNING_LOG.md` for explanations |

Final deliverables: schema and annotation guide; reproducible data pipeline; manual model and task-training path; rule baseline; saved local checkpoint; CLI; evaluation report; and a learning log covering decisions and failures.

## Later extensions

Start these only after the first release is complete. Each is a separate milestone with its own experiment record.

| Extension | Dev effort | What you build | Done when |
|---|---|---|---|
| A1. Controlled pretraining | 3–5 days dev + training | Clean, deduplicated PubMed + PMC-OA corpus under a documented reuse policy; resumable next-token pretraining of the **same** architecture and tokenizer as the first-release extractor | Pretrained-then-task-trained extractor compared with the random-init extractor using identical architecture, tokenizer, extraction data, splits, and task-training settings; pretraining compute recorded separately |
| A2. Tokenizer and scale | per experiment | Domain BPE tokenizer; larger configurations (about 125M, then 350M if justified) | Each experiment changes one variable against the best previous configuration; extraction metrics compared on identical held-out data |
| B. MIMIC adaptation | 2 days (after approval) | Continued pretraining on MIMIC-IV-Note, locally only; a separately annotated real-note evaluation set | Held-out MIMIC perplexity improves; extraction metrics reported on annotated real notes |
| C. EHR tasks | 3–5 days | ICD-10 multi-label coding from discharge summaries; LoRA-tuned radiology findings → impression | Metrics reported against GPT-2-small and published baselines |
| D. Other projects | varies | Patient timeline summarizer, FHIR mapping, terminology normalization, synthetic EHR event generator | Defined in its own plan |

### Compute sizing for pretraining (Extension A)

Training cost is about 6·N·D FLOPs, where N is parameters and D is training tokens. No durations are claimed here: wall time is planned tokens ÷ measured sustained tokens/second, plus evaluation and checkpoint overhead. The table shows the sustained throughput a target duration would require, excluding overhead.

| Model | Planned tokens | Needed for 1 day | Needed for 3 days | Measured tokens/s and peak memory |
|---|---|---|---|---|
| ~125M | 2.5B | ~29,000 tokens/s | ~9,600 tokens/s | to benchmark |
| ~350M | 3.5B | ~40,500 tokens/s | ~13,500 tokens/s | to benchmark |
| ~350M | 7B | ~81,000 tokens/s | ~27,000 tokens/s | to benchmark |

Models of 1B parameters or more remain out of scope.

Benchmark each proposed configuration at its intended sequence length, batch size, and precision: warm up, time representative optimizer steps, and record sustained tokens/second and peak memory. Throughput measured on one model size cannot be reused for another. If a configuration's measured duration is unacceptable, reduce planned tokens or model size before starting.

Multi-day run requirements:

- Checkpoints include model and optimizer state, RNG state, and data position, saved at a fixed interval; a crash loses at most the work since the last checkpoint. Test resumption against an uninterrupted run before the long run.
- Log validation loss every few hundred steps to stop bad runs early. Compare perplexity only between runs with the same tokenizer; use bits per byte when tokenizers differ.
- Freeze the tokenizer and dataset version before the long run.

### Candidate EHR tasks (Extension C and beyond)

| Task | Data | Metric |
|---|---|---|
| ICD-10 coding from discharge summaries (same classification-head pattern as the Chapter 6 classifier, but multi-label) | MIMIC-IV-Note + `diagnoses_icd` | micro/macro F1, P@k; compare with Edin et al. (2023) |
| Radiology findings → impression | MIMIC-CXR reports | ROUGE-L, F1-RadGraph, finding preservation, negation |
| PHI de-identification tagger | n2c2 2006/2014 de-id | entity F1 |
| Medication/dosage extraction → JSON | n2c2 2018 Track 2 | F1 per field |
| Text-to-SQL over an EHR schema | EHRSQL (MIMIC-III/eICU) | execution accuracy |
| Patient timeline summarizer | Synthea or MIMIC-IV structured data | correct dates, retained facts, omissions, invented events |

## Data sources

### Open, no credentialing

| Source | Data | Use and licence notes |
|---|---|---|
| [Synthea](https://synthetichealth.github.io/downloads.html) | Synthetic longitudinal patient records (FHIR, CSV, C-CDA); the Coherent Dataset includes simple notes | First-release starting point. Apache-2.0; cite the project and record the downloaded version |
| PubMed baseline (NLM FTP) | ~36M abstracts, roughly 5B tokens | Extension A candidate corpus. Some abstracts carry publisher copyright; follow the [NLM download terms](https://www.nlm.nih.gov/databases/download.html) |
| [PMC Open Access Subset](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/) | Full-text biomedical articles | Extension A. Licences vary by article; use the commercial or non-commercial subset as appropriate and the approved retrieval services |
| [DailyMed](https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm) / openFDA | Drug labels | Medication terminology; check reuse terms |
| ClinicalTrials.gov | Trial records | Check the site's terms and record them in the manifest |
| MedlinePlus | Consumer health text | Mixed: some pages are NLM-authored, others are licensed copyrighted content; select only reusable content per the [MedlinePlus usage policy](https://www.medlineplus.gov/about/using/usingcontent/) |
| CMS DE-SynPUF | Synthetic Medicare claims | Structured EHR tasks |
| PMC-Patients, Asclepius synthetic notes | Note-like case summaries | Non-commercial (CC BY-NC-SA); check each licence |
| ICD-10-CM, LOINC, RxNorm, SNOMED CT | Vocabularies | ICD-10-CM free from CMS; LOINC free licence; RxNorm and SNOMED via UMLS licence (SNOMED depends on country) |
| PubMed 200k RCT, PubMedQA, MedQA, MedMCQA | Evaluation sets | Pretraining-quality probes |

Before building any corpus, write a documented selection and reuse policy into its manifest: which records are included, the licence or terms each relies on, and how restricted items are excluded. Licence notes in this table are starting points, not clearance.

Avoid MTSamples for anything published: it is scraped and its licence is unclear.

### Credentialed (real EHR text)

| Source | Data | Access |
|---|---|---|
| [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/2.2/) | ~330k discharge summaries, ~2.3M radiology reports | PhysioNet credentialing + DUA (milestone 0) |
| [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/) | Structured hospital and ICU data | Same; notes are a separate resource |
| MIMIC-III notes, MIMIC-CXR, eICU-CRD, MedNLI | Notes, radiology reports, ICU data, clinical NLI | Same |
| n2c2/i2b2 (Harvard DBMI portal) | De-id, medication, smoking, temporal-relation shared-task data | Separate application and DUA |

Review the [access process](https://mimic.mit.edu/docs/faq/how-to-get-access.html) and [MIMIC-IV usage notes](https://physionet.org/content/mimiciv/3.1/) before planning to publish a model trained on credentialed data.

## References

- [NVIDIA DGX Spark hardware overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
- [Synthea downloads and reuse information](https://synthetichealth.github.io/downloads.html)
- [MIMIC-IV-Note access, license, and data description](https://physionet.org/content/mimic-iv-note/2.2/)
- [MIMIC-IV access and usage guidance](https://mimic.mit.edu/docs/faq/how-to-get-access.html)
- [PMC Open Access Subset terms and retrieval methods](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/)
- [DailyMed bulk label downloads](https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm)

# Data discussion and acquisition options

## Decision

Start with synthetic patient data for the first extraction prototype. Seek additional permitted biomedical text and, if feasible, credentialed clinical data after the pipeline is working and evaluated. Additional data is an option, not a dependency for the first release.

This document records the discussion and proposed sequence. No dataset has been acquired or approved through these notes, and no model performance is established.

## Two different data needs

1. **Extraction training:** clinical note paired with expected JSON facts and evidence spans. This teaches the specific task.
2. **Language-model pretraining:** unlabeled biomedical or clinical text for next-token prediction. This may improve domain familiarity but does not provide extraction labels automatically.

The current plan begins with extraction training from random weights. Domain pretraining is a later experiment, followed by extraction training and comparison on identical held-out data.

## Available options

| Source | Potential role | Access and limitations |
|---|---|---|
| [Synthea](https://synthetichealth.github.io/synthea/) | Synthetic patient records used to construct note/JSON pairs | Public starting point. Record the generator version, settings, seed, and applicable licenses. Structured records are not ready-made clinical narrative/annotation pairs. |
| Independently written synthetic notes | Schema design, edge cases, and a held-out challenge set | Write expected facts and evidence offsets explicitly. Keep development examples separate from blind evaluation. |
| [PubMed](https://pubmed.ncbi.nlm.nih.gov/) | Biomedical abstracts for possible pretraining | Check permitted retrieval methods and reuse rights. Publicly accessible abstracts are not uniformly open-licensed or automatically redistributable. |
| [PMC Open Access Subset](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/) | Full-text biomedical articles for possible pretraining | Use authorized retrieval methods and select articles whose individual licenses permit the intended use. Biomedical articles differ from clinical notes. |
| [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/) | Later real-note research, possible continued pretraining, and separately annotated evaluation | Requires credentialing and acceptance of the dataset agreement. Verify current requirements before applying. Notes do not include ready-made labels for our extraction schema. Restricted data and derived artifacts require separate handling. |
| Other licensed datasets or approved collaborations | Potential future clinical diversity and task annotations | Evaluate access, consent/de-identification, license/DUA, annotation compatibility, quality, cost, and redistribution terms before acquisition. Do not assume availability or approval. |

## Phase 1 — synthetic extraction prototype

1. Finalize the JSON schema and annotation rules using a small hand-written set.
2. Inspect a small Synthea sample before scaling generation.
3. Render structured facts into deterministic clinical paragraphs and derive labels from what each paragraph actually says.
4. Deliberately construct negation, uncertainty, historical context, and family-attribution examples. Missing source fields do not establish negative findings.
5. Validate evidence text and exact character offsets; leave unstated medication details null.
6. Split by patient before generating examples; deduplicate across splits and reserve template families for generalization tests.
7. Maintain an independently written challenge set, separate from the examples used during development.
8. Train and compare the model against a rule baseline. Report synthetic-task performance without real-clinical claims.

The root `example_01.txt` through `example_10.txt` files are illustrative development material, not actual model output or a reserved test set. Review them against the finalized annotation guide before any reuse.

## Phase 2 — permitted biomedical pretraining data

After the extraction baseline works, investigate PubMed and the PMC Open Access Subset. Start with a small, documented, license-compatible sample rather than a bulk download.

Record provenance and rights, clean and deduplicate text, exclude evaluation material, and measure tokenizer behavior and compute requirements. Compare pretraining plus extraction training against the original baseline on unchanged held-out sets. Track the additional data and compute budget.

Do not assume that biomedical publication text represents clinical writing or that more text necessarily improves extraction.

## Phase 3 — procure additional clinical data if feasible

Investigate PhysioNet credentialing and MIMIC-IV-Note access early if desired; approval can proceed in parallel with synthetic development. Other sources or collaborations can be considered when there is a clear experimental need.

Before downloading restricted data:

- Confirm the current access requirements, permitted uses, retention rules, and publication/redistribution restrictions.
- Establish storage outside the repository and verify that online coding agents cannot read it through either file tools or shell processes. Test controls with harmless dummy files first.
- Decide how annotations and domain review will be obtained; real notes are not automatically ground truth for this schema.
- Define an independent evaluation protocol and account for domain shift.
- Review restrictions on derived datasets, model checkpoints, and example outputs before sharing anything publicly.

If access or safe handling is not feasible, continue with permitted synthetic/public sources. Do not weaken the controls just to obtain more data.

## Storage, Git, and PostgreSQL

- Permitted local raw data: `llm/data/raw/`.
- Intermediate transformations: `llm/data/interim/`.
- Prepared datasets: `llm/data/processed/{pretraining,extraction}/{train,validation,test}/`.
- Local weights: `llm/checkpoints/`; generated run artifacts: `llm/experiments/runs/`.
- Credentialed data and restricted derived artifacts: outside the repository under verified access controls.
- PostgreSQL: application records, extraction results, and model-version references—not training corpora or weight binaries.

The data/artifact folders are ignored by Git, apart from directory placeholders. `.gitignore` does not grant data rights, block agent reads, or remove already tracked files or historical commits. Inspect staged changes before committing. Avoid adding sensitive content in the first place; any accidental disclosure requires appropriate incident handling, not merely a new ignore rule.

## Manifest and sharing rules

For each dataset, keep a safe manifest with source URL, license/DUA, version, retrieval date, hashes, generation settings where applicable, preprocessing, split strategy, and intended use. Record patient/example counts separately from total tokens processed.

The repository's MIT License covers project software and associated documentation, not third-party data rights. Commit safe code, configurations, manifests, tests, and aggregate summaries. Publish only data and model artifacts whose terms permit distribution; do not assume locally trained weights are unrestricted.

Use synthetic notes for the Next.js/FastAPI showcase. Synthetic success is evidence about the synthetic task, not clinical reliability.

See also [data policy](DATA_POLICY.md), [architecture](ARCHITECTURE.md), and the [development plan](../ANAMNESISLM_PLAN.md).

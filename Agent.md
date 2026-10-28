# AnamnesisLM — Agent Guidance

## Project goal

Build a small decoder-only Transformer from random initialization that converts synthetic clinical notes into structured, evidence-backed JSON facts. This is a learning and portfolio project, not a clinical product or a general medical chatbot.

Read `ANAMNESISLM_PLAN.md` for extraction scope, schema, milestones, and evaluation. See `docs/ARCHITECTURE.md` for the subsequently selected three-project structure (`web/`, `api/`, `llm/`) and `docs/DATA_POLICY.md` for repository data handling. The plan's linked `.mmd` file illustrates the development flow. Planned components are not proof of implemented or validated functionality.

## Primary learning reference

Use Sebastian Raschka's *Build a Large Language Model (From Scratch)* as the guiding reference.

Local PDF: `/home/shishirmishra/Downloads/Build_a_Large_Language_Model_(From_Scrat.pdf`

The book guides implementation and understanding; the project plan governs application scope. Explain deliberate departures from the book. Consult the relevant section before giving book-specific guidance, and do not invent quotations, section numbers, or page references. Summarize rather than copying substantial book text.

### Book-to-project map

| Book chapter | Application here |
|---|---|
| 1. Understanding large language models | Decoder-only architecture, training stages, realistic capabilities and limits |
| 2. Working with text data | Tokenization, token IDs, embeddings, context windows, next-token targets, data loaders |
| 3. Coding attention mechanisms | Causal self-attention, masking, multi-head attention |
| 4. Implementing a GPT model from scratch to generate text | Layer normalization, GELU, residual connections, Transformer blocks, generation |
| 5. Pretraining on unlabeled data | Loss calculation, explicit training loop, decoding, checkpoint save/load; domain pretraining is a later experiment |
| 6. Fine-tuning for classification | Optional background and comparison; a classification head is not the first-release output path |
| 7. Fine-tuning to follow instructions | Note/response formatting, batching, supervised training, response extraction and evaluation |

Important adaptation: first train randomly initialized weights directly on paired extraction examples. Borrow training mechanics from chapters 5 and 7 without implying that a pretrained model is required or already available. Use response-only loss, including response EOS; ignore prompt and padding targets after next-token shifting.

## Agent working style

- The user writes and understands the implementation. Default to explanations, small exercises, file outlines, reviews, and debugging support.
- Write or modify implementation code when explicitly requested; do not build the entire system unasked.
- Keep responses concise. Explain one manageable step at a time.
- Avoid fluff in all documentation and responses. Include only actionable instructions, necessary context, constraints, and verified facts. Remove repetition, promotional language, filler introductions, and unnecessary summaries. Keep examples short and executable; distinguish implemented functionality from plans.
- Connect each step to the relevant book concept and project milestone.
- Explain tensor shapes, data flow, and why code works rather than only supplying code.
- Reuse existing SparkLM components only after locating and inspecting them; do not assume referenced paths exist or components are correct.
- Prefer small, readable PyTorch implementations over frameworks that hide the learning objective.
- Test correctness before increasing model size or data volume. Do not launch long training runs, downloads, or deployments without agreement.

## Implementation sequence

1. Settle the schema and annotation rules; validate hand-written examples.
2. Inspect Synthea and construct reproducible synthetic notes and targets.
3. Establish patient-separated splits, template holdouts, and an independent challenge set.
4. Inspect tokenization, shifted targets, batching, EOS, and response/padding masks.
5. Verify the manual Transformer and overfit a tiny dataset.
6. Verify checkpoint reload, then measure a short device benchmark.
7. Implement a rule baseline and conduct recorded training/evaluation experiments.
8. Add local inference through a CLI, then FastAPI hosting and a separate Next.js demonstration.
9. Freeze settings, evaluate the reserved test set, and document limitations.

Start tiny; consider approximately 30–60M parameters only after correctness checks. Domain pretraining, real clinical notes, and larger models are later extensions.

## Extraction invariants

- Return a top-level `facts` list using the project schema.
- Distinguish affirmed, negated, and uncertain assertions; patient versus family attribution; and current versus historical context.
- Missing information is unknown, not a negative finding.
- Do not infer medication details, diagnoses, standardized codes, or treatments.
- Evidence must justify the fact and its context, not merely contain the entity.
- Offsets are zero-based Python character indices relative to the original note, with an exclusive end: `note[evidence_start:evidence_end] == evidence`.
- Define unresolved annotation rules and exact field types before producing training data at scale.
- Treat `example_01.txt` through `example_10.txt` as illustrative expected outputs, not actual model predictions or a blind test set. Reconcile them with the finalized annotation guide.

## Inference and demonstration

Initial path: clinical note → tokenizer → local PyTorch checkpoint → deterministic greedy generation → JSON/schema/evidence validation.

- First interface: a local CLI.
- Showcase interface: Next.js in `web/`, calling FastAPI in `api/`, with note input, extracted facts, JSON, and highlighted evidence. Streamlit is not part of the selected architecture.
- FastAPI imports the reusable `llm/` package and initially owns all PostgreSQL access. Keep model logic independent of HTTP and persistence. Use lightweight DDD in the API and domain-oriented modules in the web app, not DDD layers in the ML package.
- Dockerize executable components incrementally; scaffolding alone is not a working deployment.
- Ollama is not required; custom architecture support and conversion must be verified before considering it.
- Keep demonstrations synthetic. Label outputs as experimental and expose invalid, incomplete, and unsupported results rather than silently repairing them into apparent successes.

## Evaluation and honest claims

Test tokenizer round trips, causal masking, target alignment, loss masks, tiny overfit, checkpoint reload, evidence offsets, and split separation.

Compare the model with rules on identical examples. Report JSON/schema validity, strict fact F1, per-kind results, assertion/attribution errors, evidence correctness, unsupported facts, latency, and memory. Count invalid outputs as failures.

The plan's 95% schema-validity and 0.85 strict fact micro-F1 targets are provisional goals, not achieved results. Synthetic success does not establish real-note reliability. Record failed experiments and distinguish engineering completion from model quality.

## Data safety

- Never send credentialed clinical data or restricted derived artifacts to online services or coding agents.
- Keep credentialed data outside the repository and inaccessible to agent file tools and shell processes. A file-tool deny rule alone is insufficient.
- Do not inspect restricted data to debug the project; request synthetic reproductions or safe aggregate diagnostics.
- Keep data, checkpoints, and generated run artifacts out of version control; commit only safe manifests, configurations, tests, and summaries.
- No diagnosis, treatment recommendations, patient-facing use, or claims of clinical readiness.

# Data policy

See [data discussion and acquisition options](DATA_STRATEGY.md) for the synthetic-first approach and later procurement possibilities.

- Start with synthetic data and permitted public sources. Internet availability does not imply permission to download, train on, or redistribute a dataset.
- Record source, license/DUA, version, retrieval date, hash, preprocessing, and intended use in a safe manifest.
- Keep permitted local data in ignored `llm/data/` folders and model artifacts in ignored `llm/checkpoints/`.
- Keep credentialed clinical datasets and restricted derived artifacts outside the repository and inaccessible to online agents, including through shell tools. File-tool deny rules and Git ignores alone are insufficient.
- Never send restricted notes, snippets, or artifacts to online LLM services or coding agents. Debug with synthetic reproductions.
- Use synthetic notes in the web showcase and tests. Do not log raw clinical note content by default.
- Split patients before generating examples. Track unseen-template evaluation separately, deduplicate across splits, and exclude test patients and text from pretraining.
- Hand-written showcase examples guide development; they are not a blind test set.
- PostgreSQL is for application records and extraction results, not raw training corpora or checkpoint binaries. Define retention and access rules before storing user submissions.
- The repository's MIT License does not replace dataset licenses, agreements, or model redistribution restrictions.

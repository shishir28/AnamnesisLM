"use client";

import { useEffect, useState, type FormEvent } from "react";

import { evidenceSegments, type ExtractionResult, type ModelInfo } from "../domain/extraction";
import { extractFacts, getModelInfo } from "../infrastructure/client";
import FactTable from "./FactTable";

const examples = [
  {
    name: "Negation & family",
    note: "Patient is a 54-year-old male. Currently has Asthma. The patient denies any chest pain. Mother has Diabetes mellitus type 2.",
  },
  {
    name: "Medication & allergy",
    note: "Patient is a 38-year-old female. Takes Metformin. Allergic to Penicillin. History of Pneumonia.",
  },
  {
    name: "Uncertainty",
    note: "Patient is a 67-year-old male. Currently has Hypertension. Possible shortness of breath. No fever.",
  },
];

type View = "facts" | "json" | "raw";

export default function ExtractionLab() {
  const [note, setNote] = useState(examples[0].note);
  const [maxTokens, setMaxTokens] = useState(400);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [statusError, setStatusError] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<ExtractionResult | null>(null);
  const [submittedNote, setSubmittedNote] = useState("");
  const [view, setView] = useState<View>("facts");

  async function refreshModel() {
    setStatusError("");
    try {
      setModel(await getModelInfo());
    } catch (reason) {
      setModel(null);
      setStatusError(reason instanceof Error ? reason.message : "Could not fetch model status.");
    }
  }

  useEffect(() => {
    void refreshModel();
  }, []);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError("");
    setResult(null);
    setSubmittedNote(note);
    try {
      setResult(await extractFacts(note, maxTokens));
      setView("facts");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Extraction failed.");
    } finally {
      setBusy(false);
    }
  }

  const facts = result?.facts ?? [];
  const segments = evidenceSegments(submittedNote, facts);

  return (
    <div className="clinical-app">
      <header className="topbar">
        <a className="brand" href="/">
          <span className="brand-symbol" aria-hidden="true">
            +
          </span>
          AnamnesisLM
        </a>
        <span className="header-divider" aria-hidden="true" />
        <span className="header-label">Clinical documentation workspace</span>
        <span className="environment-label">SYNTHETIC DATA / RESEARCH</span>
      </header>

      <div className="app-body">
        <aside className="sidebar" aria-label="Workspace navigation">
          <p className="sidebar-title">WORKSPACE</p>
          <nav>
            <a href="#note-editor" className="nav-item">
              <span aria-hidden="true">01</span> Note entry
            </a>
            <a href="#model-output" className="nav-item">
              <span aria-hidden="true">02</span> Extraction review
            </a>
            <a href="#model-service" className="nav-item">
              <span aria-hidden="true">03</span> Model status
            </a>
          </nav>
          <div className="sidebar-footer">
            <strong>Research environment</strong>
            <span>
              Not connected to an EHR.
              <br />
              No patient records stored.
            </span>
          </div>
        </aside>

        <main className="shell">
          <div className="page-heading">
            <div>
              <p className="breadcrumb">Clinical documentation / Extraction review</p>
              <h1>Clinical note review</h1>
              <p className="page-description">
                Review model-generated findings against the source note.
              </p>
            </div>
            <span className="session-label">UNSAVED SESSION</span>
          </div>

          <div className="document-context" aria-label="Document context">
            <div>
              <span>Document</span>
              <strong>Synthetic clinical note</strong>
            </div>
            <div>
              <span>Review type</span>
              <strong>Structured fact extraction</strong>
            </div>
            <div>
              <span>Persistence</span>
              <strong>Not stored</strong>
            </div>
          </div>

          <aside className="safety">
            <strong>Research use only.</strong> Do not enter real patient information. Model output
            is not validated for clinical use.
          </aside>

          <section id="model-service" className="modelbar" aria-label="Model status">
            <span className={`status-dot ${model?.ready ? "ready" : ""}`} />
            <strong>
              {model?.ready
                ? "Model ready"
                : model
                  ? "Model unavailable"
                  : statusError
                    ? "API unavailable"
                    : "Checking connection"}
            </strong>
            <span className="model-detail">
              {model?.ready
                ? `${model.model_version} · ${model.device} · ${model.context_length}-token context`
                : statusError || model?.error || "Connecting to FastAPI…"}
            </span>
            <button type="button" className="secondary compact" onClick={() => void refreshModel()}>
              Refresh status
            </button>
          </section>

          <div className="workspace">
            <section id="note-editor" className="panel input-panel">
              <div className="panel-heading">
                <h2>Clinical note</h2>
                <span className="section-label">SOURCE DOCUMENT</span>
              </div>
              <form onSubmit={submit}>
                <label className="field-label" htmlFor="note">
                  Choose a synthetic example or write your own
                </label>
                <div className="examples">
                  {examples.map((example) => (
                    <button
                      type="button"
                      className="secondary"
                      key={example.name}
                      disabled={busy}
                      onClick={() => setNote(example.note)}
                    >
                      {example.name}
                    </button>
                  ))}
                </div>
                <div className="editor-header">
                  <span>Note text</span>
                  <span>Plain text</span>
                </div>
                <textarea
                  id="note"
                  value={note}
                  onChange={(event) => setNote(event.target.value)}
                  maxLength={12000}
                  rows={13}
                  required
                  disabled={busy}
                  spellCheck={false}
                />
                <div className="input-meta">
                  <span>{Array.from(note).length} characters</span>
                  <span>No autosave</span>
                </div>
                <div className="controls">
                  <label htmlFor="tokens">
                    Max output tokens
                    <input
                      id="tokens"
                      type="number"
                      min={1}
                      max={1024}
                      step={1}
                      value={maxTokens}
                      required
                      disabled={busy}
                      onChange={(event) => setMaxTokens(Number(event.target.value))}
                    />
                  </label>
                  <button
                    className="primary"
                    disabled={busy || !model?.ready || !note.trim()}
                    type="submit"
                  >
                    {busy ? "Extracting…" : "Extract facts"}
                  </button>
                </div>
                <p className="hint">
                  Greedy decoding. Output length may be reduced to preserve the full note within the
                  model context.
                </p>
              </form>
              {error && (
                <div className="alert error" role="alert">
                  {error}
                </div>
              )}
            </section>

            <section id="model-output" className="panel output-panel" aria-busy={busy}>
              <div className="panel-heading">
                <h2>Extraction review</h2>
                <span className="section-label">MODEL-GENERATED / UNVERIFIED</span>
              </div>
              <div aria-live="polite">
                {busy && (
                  <div className="empty">
                    <div className="pulse" />
                    <h3>Extraction in progress</h3>
                    <p>Generating structured findings from the submitted note.</p>
                  </div>
                )}
                {!busy && !result && (
                  <div className="empty">
                    <h3>No extraction results</h3>
                    <p>
                      Enter a synthetic note and select Extract facts. Review each finding and its
                      supporting evidence before drawing conclusions.
                    </p>
                  </div>
                )}
              </div>
              {!busy && result && (
                <>
                  <div className="result-meta">
                    <span className={`result-badge ${result.valid ? "pass" : "fail"}`}>
                      {result.valid ? "Schema & offsets valid" : "Validation failed"}
                    </span>
                    <span>
                      {(result.elapsed_ms / 1000).toFixed(2)} s · {result.generation_token_limit}{" "}
                      token budget
                    </span>
                  </div>
                  <p className="hint">
                    {result.model_version}. Validation checks structure and copied spans, not
                    clinical accuracy or completeness.
                  </p>
                  {result.warnings.map((warning, index) => (
                    <div className="alert" key={index}>
                      {warning}
                    </div>
                  ))}
                  {result.validation_errors.length > 0 && (
                    <div className="alert error" role="alert">
                      <strong>Output was not repaired.</strong>
                      <ul>
                        {result.validation_errors.map((message, index) => (
                          <li key={index}>{message}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  <div className="tabs" role="group" aria-label="Output view">
                    {(["facts", "json", "raw"] as const).map((tab) => (
                      <button
                        key={tab}
                        type="button"
                        aria-pressed={view === tab}
                        className={view === tab ? "active" : ""}
                        onClick={() => setView(tab)}
                      >
                        {tab === "facts"
                          ? `Facts (${facts.length})`
                          : tab === "json"
                            ? "Parsed JSON"
                            : "Raw output"}
                      </button>
                    ))}
                  </div>
                  {view === "facts" && (
                    <>
                      {facts.length > 0 && <FactTable facts={facts} note={submittedNote} />}
                      {facts.length === 0 && (
                        <p className="hint">
                          {result.facts === null
                            ? "The response could not be parsed. Inspect raw output."
                            : "No facts returned. This does not establish that the note contains none."}
                        </p>
                      )}
                    </>
                  )}
                  {view === "json" && <pre>{JSON.stringify({ facts: result.facts }, null, 2)}</pre>}
                  {view === "raw" && <pre>{result.raw_output || "(empty output)"}</pre>}
                  <div className="evidence">
                    <h3>Supporting text</h3>
                    <p>
                      {segments.map((segment, index) =>
                        segment.highlighted ? (
                          <mark key={index}>{segment.text}</mark>
                        ) : (
                          <span key={index}>{segment.text}</span>
                        ),
                      )}
                    </p>
                    <small>
                      Highlighting indicates an exact text/offset match, not verified clinical
                      support.
                    </small>
                  </div>
                </>
              )}
            </section>
          </div>
          <footer>
            <span>AnamnesisLM · Local model inference</span>
            <span>Research prototype · No clinical decisions</span>
          </footer>
        </main>
      </div>
    </div>
  );
}

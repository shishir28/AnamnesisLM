import { Fragment } from "react";

import { matchesEvidence, type Fact } from "../domain/extraction";

export default function FactTable({ facts, note }: { facts: Fact[]; note: string }) {
  return (
    <div className="findings-table-scroll" tabIndex={0} aria-label="Extracted findings table">
      <table className="findings-table">
        <thead>
          <tr>
            <th scope="col">Finding / mention</th>
            <th scope="col">Type</th>
            <th scope="col">Assertion</th>
            <th scope="col">Subject</th>
            <th scope="col">Time</th>
            <th scope="col">Evidence</th>
          </tr>
        </thead>
        <tbody>
          {facts.map((fact, index) => {
            const matched = matchesEvidence(note, fact);
            return (
              <Fragment key={index}>
                <tr>
                  <th scope="row">{String(fact.mention ?? "Missing mention")}</th>
                  <td>{String(fact.kind ?? "unknown")}</td>
                  <td>
                    <span className={`assertion ${fact.assertion === "negated" ? "negated" : ""}`}>
                      {String(fact.assertion ?? "missing")}
                    </span>
                  </td>
                  <td>{String(fact.experiencer ?? "missing")}</td>
                  <td>{String(fact.temporality ?? "missing")}</td>
                  <td>
                    <span className={matched ? "span-valid" : "span-invalid"}>
                      {matched ? "Matched" : "Mismatch"}
                    </span>
                  </td>
                </tr>
                <tr className="finding-detail">
                  <td colSpan={6}>
                    <details>
                      <summary>
                        Supporting text · [{String(fact.evidence_start ?? "?")},{" "}
                        {String(fact.evidence_end ?? "?")})
                      </summary>
                      <blockquote>{String(fact.evidence ?? "No evidence returned")}</blockquote>
                      {fact.kind === "medication" && (
                        <dl className="medication">
                          {["dose", "unit", "route", "frequency"].map((field) => (
                            <div key={field}>
                              <dt>{field}</dt>
                              <dd>{fact[field] == null ? "Unstated" : String(fact[field])}</dd>
                            </div>
                          ))}
                        </dl>
                      )}
                    </details>
                  </td>
                </tr>
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

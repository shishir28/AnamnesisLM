import { describe, expect, it } from "vitest";

import { evidenceSegments, matchesEvidence } from "../src/modules/extraction/domain/extraction";

const span = (start: number, end: number, evidence: string) => ({
  evidence_start: start,
  evidence_end: end,
  evidence,
});

describe("evidence matching", () => {
  it("matches exact spans", () => {
    expect(matchesEvidence("No fever.", span(0, 9, "No fever."))).toBe(true);
  });

  it("uses Python code point offsets for Unicode", () => {
    const note = "🙂 No fever.";
    expect(matchesEvidence(note, span(2, 11, "No fever."))).toBe(true);
    expect(evidenceSegments(note, [span(2, 11, "No fever.")])).toEqual([
      { text: "🙂 ", highlighted: false },
      { text: "No fever.", highlighted: true },
    ]);
  });

  it.each([
    span(-1, 9, "No fever."),
    span(0, 100, "No fever."),
    span(0, 9, "Wrong text"),
    span(0, 0, ""),
    span(0.5, 9, "No fever."),
    { evidence_start: "0", evidence_end: 9, evidence: "No fever." },
  ])("rejects invalid evidence: %j", (fact) => {
    expect(matchesEvidence("No fever.", fact)).toBe(false);
  });

  it("merges overlapping evidence without duplicating text", () => {
    expect(
      evidenceSegments("No fever or cough.", [
        span(0, 8, "No fever"),
        span(3, 17, "fever or cough"),
      ]),
    ).toEqual([
      { text: "No fever or cough", highlighted: true },
      { text: ".", highlighted: false },
    ]);
  });

  it("does not highlight malformed spans", () => {
    expect(evidenceSegments("No fever.", [span(0, 9, "Invented")])).toEqual([
      { text: "No fever.", highlighted: false },
    ]);
  });
});

export type Fact = Record<string, unknown>;

export interface ExtractionResult {
  raw_output: string;
  facts: Fact[] | null;
  model_version: string;
  valid: boolean;
  validation_errors: string[];
  warnings: string[];
  elapsed_ms: number;
  generation_token_limit: number;
}

export interface ModelInfo {
  ready: boolean;
  error?: string;
  model_version?: string;
  device?: string;
  context_length?: number;
  max_new_tokens?: number;
}

export interface EvidenceSegment {
  text: string;
  highlighted: boolean;
}

export function matchesEvidence(note: string, fact: Fact): boolean {
  const { evidence_start: start, evidence_end: end, evidence } = fact;
  const chars = Array.from(note); // Python character offsets, not JS UTF-16 indices.
  return (
    typeof start === "number" &&
    Number.isInteger(start) &&
    typeof end === "number" &&
    Number.isInteger(end) &&
    typeof evidence === "string" &&
    start >= 0 &&
    end > start &&
    end <= chars.length &&
    chars.slice(start, end).join("") === evidence
  );
}

export function evidenceSegments(note: string, facts: Fact[]): EvidenceSegment[] {
  const chars = Array.from(note);
  const mask = new Array<boolean>(chars.length).fill(false);
  for (const fact of facts) {
    if (!matchesEvidence(note, fact)) continue;
    for (let i = fact.evidence_start as number; i < (fact.evidence_end as number); i++) {
      mask[i] = true;
    }
  }
  const segments: EvidenceSegment[] = [];
  chars.forEach((char, index) => {
    const previous = segments.at(-1);
    if (previous && previous.highlighted === mask[index]) previous.text += char;
    else segments.push({ text: char, highlighted: mask[index] });
  });
  return segments;
}

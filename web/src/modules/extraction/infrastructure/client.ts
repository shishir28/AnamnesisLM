import type { ExtractionResult, ModelInfo } from "../domain/extraction";

async function readResponse<T>(response: Response): Promise<T> {
  let body;
  try {
    body = await response.json();
  } catch {
    throw new Error("The server returned an unreadable response.");
  }
  if (!response.ok) {
    const detail = body.detail;
    if (typeof detail === "string") throw new Error(detail);
    if (Array.isArray(detail)) {
      throw new Error(detail.map((item) => item.msg ?? "Invalid request").join(" "));
    }
    throw new Error(`Request failed (${response.status}).`);
  }
  return body as T;
}

export async function getModelInfo(): Promise<ModelInfo> {
  return readResponse<ModelInfo>(await fetch("/api/model", { cache: "no-store" }));
}

export async function extractFacts(note: string, maxNewTokens: number): Promise<ExtractionResult> {
  const response = await fetch("/api/extract", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ note, max_new_tokens: maxNewTokens }),
  });
  return readResponse<ExtractionResult>(response);
}

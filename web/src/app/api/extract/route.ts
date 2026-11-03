import { NextResponse } from "next/server";

import { forwardToApi } from "@/shared/server/api";

export const runtime = "nodejs";

export async function POST(request: Request) {
  // Bound bytes before parsing or proxying; never log note content.
  if (Number(request.headers.get("content-length")) > 65536) {
    return NextResponse.json({ detail: "Request is too large." }, { status: 413 });
  }
  const reader = request.body?.getReader();
  if (!reader) return NextResponse.json({ detail: "Request body required." }, { status: 400 });
  const chunks: Uint8Array[] = [];
  let size = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.byteLength;
    if (size > 65536) {
      await reader.cancel();
      return NextResponse.json({ detail: "Request is too large." }, { status: 413 });
    }
    chunks.push(value);
  }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return forwardToApi("/extract", new TextDecoder().decode(bytes));
}

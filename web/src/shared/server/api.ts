import "server-only";

import { NextResponse } from "next/server";

export async function forwardToApi(path: "/model" | "/extract", body?: string) {
  const baseUrl = process.env.API_BASE_URL ?? "http://127.0.0.1:8000";
  try {
    const response = await fetch(new URL(path, baseUrl), {
      method: body === undefined ? "GET" : "POST",
      headers: { "Content-Type": "application/json" },
      body,
      cache: "no-store",
      signal: AbortSignal.timeout(180_000),
    });
    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch {
    return NextResponse.json(
      { detail: "API unavailable or inference timed out. Check the API service and model status." },
      { status: 502 },
    );
  }
}

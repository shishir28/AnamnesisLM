import { forwardToApi } from "@/shared/server/api";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  return forwardToApi("/model");
}

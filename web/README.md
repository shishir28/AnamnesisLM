# Web

Next.js App Router interface for synthetic-note extraction. The EMR-style workspace uses a compact navy/gray layout and a findings table with expandable evidence. It shows model status, raw/parsed output, schema errors, and matching evidence spans. It is not connected to an EHR.

## Docker

From the repository root:

```bash
docker compose up --build -d api web
```

Open http://localhost:3000. The server proxies `/api/model` and `/api/extract` to FastAPI at `http://api:8000`. No browser CORS setup or database connection is required.

For a remote SSH host, forward the web port from your local computer:

```bash
ssh -N -L 3000:127.0.0.1:3000 user@host
```

Then open http://localhost:3000 locally. Add `-L 8000:127.0.0.1:8000` to access API docs as well.

## Native development

Requires Node.js 22+ and the API running on port 8000:

```bash
cd web
npm ci
cp .env.example .env.local
npm run dev
```

`API_BASE_URL` is server-side configuration, not a `NEXT_PUBLIC_` variable. Docker supplies it at runtime. Do not place secrets in browser-exposed configuration.

## Tests

```bash
cd web
npm test
npm run typecheck
npm run build

# Browser tests require the web app running on port 3000.
npx playwright install chromium
npx playwright test
```

Vitest covers evidence offsets, Unicode, overlapping spans, and invalid evidence. Playwright covers successful display, invalid model output, unavailable models, and request failures using mocked API responses. Override the running app URL with `PLAYWRIGHT_BASE_URL`.

## Structure

- `src/app/`: pages, layout, and same-origin server proxy routes.
- `src/modules/extraction/domain/`: response types and evidence matching.
- `src/modules/extraction/infrastructure/`: HTTP client.
- `src/modules/extraction/presentation/`: interactive extraction lab.
- `src/shared/server/`: server-only FastAPI forwarding.

No notes are stored by the app. React escapes generated text; no raw model HTML is rendered. Evidence highlighting uses Python-compatible Unicode character offsets. Structural validation does not establish clinical correctness. A 180-second proxy timeout does not cancel generation already running in FastAPI.

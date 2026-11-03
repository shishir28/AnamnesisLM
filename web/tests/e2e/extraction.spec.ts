import { expect, test } from "@playwright/test";

const model = {
  ready: true,
  model_version: "test@step1",
  device: "cpu",
  context_length: 512,
};

const note = "Patient denies chest pain.";
const fact = {
  kind: "symptom",
  mention: "chest pain",
  assertion: "negated",
  experiencer: "patient",
  temporality: "unspecified",
  evidence: note,
  evidence_start: 0,
  evidence_end: note.length,
  dose: null,
  unit: null,
  route: null,
  frequency: null,
};

const output = {
  raw_output: JSON.stringify({ facts: [fact] }),
  facts: [fact],
  model_version: "test@step1",
  valid: true,
  validation_errors: [],
  warnings: [],
  elapsed_ms: 100,
  generation_token_limit: 400,
};

test.beforeEach(async ({ page }) => {
  await page.route("**/api/model", (route) => route.fulfill({ json: model }));
});

test("submits a note and displays facts, evidence, JSON and raw output", async ({ page }) => {
  await page.route("**/api/extract", async (route) => {
    expect(route.request().postDataJSON()).toEqual({ note, max_new_tokens: 400 });
    await route.fulfill({ json: output });
  });
  await page.goto("/");
  await expect(page.getByText("Model ready", { exact: true })).toBeVisible();
  await page.getByLabel("Choose a synthetic example or write your own").fill(note);
  await page.getByRole("button", { name: "Extract facts" }).click();
  await expect(page.getByText("Schema & offsets valid", { exact: true })).toBeVisible();
  await expect(page.locator("mark")).toHaveText(note);
  await expect(page.getByText("chest pain", { exact: true })).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "Assertion", exact: true })).toBeVisible();
  await expect(page.getByRole("cell", { name: "Matched", exact: true })).toBeVisible();
  await page.locator("summary").first().click();
  await expect(page.locator("blockquote")).toHaveText(note);
  await page.getByRole("button", { name: "Parsed JSON" }).click();
  await expect(page.locator("pre")).toContainText('"assertion": "negated"');
  await page.getByRole("button", { name: "Raw output", exact: true }).click();
  await expect(page.locator("pre")).toHaveText(output.raw_output);
});

test("invalid generation remains visible and is not repaired", async ({ page }) => {
  await page.route("**/api/extract", (route) =>
    route.fulfill({
      json: {
        ...output,
        facts: null,
        raw_output: "{broken",
        valid: false,
        validation_errors: ["Invalid JSON"],
      },
    }),
  );
  await page.goto("/");
  await page.getByRole("button", { name: "Extract facts" }).click();
  await expect(page.getByText("Validation failed", { exact: true })).toBeVisible();
  await expect(
    page.getByRole("alert").filter({ hasText: "Output was not repaired." }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Raw output", exact: true }).click();
  await expect(page.locator("pre")).toHaveText("{broken");
});

test("unavailable model disables extraction", async ({ page }) => {
  await page.route("**/api/model", (route) =>
    route.fulfill({
      json: { ready: false, error: "Model unavailable. Check the checkpoint." },
    }),
  );
  await page.goto("/");
  await expect(page.getByText("Model unavailable", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Extract facts" })).toBeDisabled();
});

test("request failures are displayed without a fabricated result", async ({ page }) => {
  await page.route("**/api/extract", (route) =>
    route.fulfill({
      status: 429,
      json: { detail: "Model is busy. Retry later." },
    }),
  );
  await page.goto("/");
  await page.getByRole("button", { name: "Extract facts" }).click();
  await expect(page.getByRole("alert").filter({ hasText: "Model is busy." })).toHaveText(
    "Model is busy. Retry later.",
  );
  await expect(page.getByText("Schema & offsets valid", { exact: true })).not.toBeVisible();
});

# Task 2 report — upload and live processing pages

## Scope

Added the Russian upload flow, SSE-backed processing view, `/processing/[videoId]` route, and the home-page integration. The interface retains the existing inspection-console field language: blueprint structure, mineral surfaces, amber progress signal, and concise operational copy.

## TDD evidence

### Red

Created `upload-form.test.tsx` and `processing-status.test.tsx`, then ran:

`cd web && npm test -- upload-form.test.tsx processing-status.test.tsx`

The run failed as expected because `@/components/upload-form` and `@/components/processing-status` did not exist.

### Green

Implemented the smallest client components that call the typed API client, open and close the status EventSource, and use App Router navigation. The focused suite then passed: 2 files, 4 tests.

## Verification

| Command | Result |
| --- | --- |
| `cd web && npm test -- upload-form.test.tsx processing-status.test.tsx` | Passed: 4/4 tests. |
| `cd web && npm test` | Passed: 11/11 tests. |
| `cd web && npm run build` | Passed outside the sandbox: Next.js 16.1.6 compiled and emitted `/` plus dynamic `/processing/[videoId]`. |

The first build attempt was blocked because the sandbox prevents Turbopack from binding its internal worker port; the approved outside-sandbox rebuild passed.

## Concerns

- The project runs Vitest 4 under Node 23.7.0, while Vitest declares Node 20, 22, or 24+ support. Tests passed, but CI should use a supported LTS release.
- `npm install` reported three high-severity transitive dependency advisories. No automated remediation was applied because it may change the dependency graph.

## Review round 1 remediation

- Treat both backend terminal statuses as terminal SSE events: close the `EventSource` for `done` and `failed`, while routing only on `done`. A failed server result therefore remains visible without a misleading connection-retry control.
- Separate upload and analysis-start failures. After an upload succeeds, retrying a failed analysis start reuses its stored video ID; changing the file, project, or media type clears that pending ID.

### TDD and verification

- Added terminal-failure and analysis-retry regressions first. Both failed against the previous behavior: the failed status exposed retry UI, and an analysis-start failure was labelled as an upload failure.
- `cd web && npm test -- upload-form.test.tsx processing-status.test.tsx` — 6/6 passed.
- `cd web && npm test` — 13/13 passed.
- `cd web && npx tsc --noEmit` — passed after the production build completed.
- `cd web && npm run build` — passed outside the sandbox; Turbopack requires an internal worker port.

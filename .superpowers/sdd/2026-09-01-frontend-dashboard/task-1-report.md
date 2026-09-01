# Task 1 report — foundation and typed backend client

## Scope

Created the standalone `web/` Next.js App Router foundation and typed FastAPI client. The initial page establishes the Russian construction-inspection console visual system: mineral paper framing, blueprint structure, amber signal state, restrained condensed utility typography, and a measurement-grid texture limited to the page frame.

## TDD evidence

### Red

1. Created the test harness and `web/tests/api.test.ts` before `web/lib/api.ts`.
2. Ran `cd web && npm test -- api.test.ts`.
3. Result: failed as expected with `Cannot find package '@/lib/api' imported from .../web/tests/api.test.ts` (0 tests collected). This proves the test was exercising the absent client boundary.

### Green

1. Added the minimal `api` wrappers and backend-shaped TypeScript types.
2. Ran `cd web && npm test -- api.test.ts`.
3. Result: 1 test file passed; 6 tests passed. Coverage includes the requested video upload behaviour, photo upload endpoint, analysis start, report fetch, video history fetch, and URL construction.

## Commands and results

| Command | Result |
| --- | --- |
| `cd web && npm test -- api.test.ts` (red) | Failed as expected: `@/lib/api` absent. |
| `cd web && npm test -- api.test.ts` (green) | Passed: 6/6 tests. |
| `cd web && npm run build` (sandbox) | Blocked by sandbox policy: Turbopack worker could not bind a local port. |
| `cd web && npm run build` (approved outside sandbox) | Passed: Next.js 16.1.6 completed a production build; `/` prerendered as static. |

## Files changed

- `web/package.json`
- `web/tsconfig.json`
- `web/next.config.ts`
- `web/postcss.config.mjs`
- `web/tailwind.config.ts`
- `web/vitest.config.ts`
- `web/app/layout.tsx`
- `web/app/globals.css`
- `web/app/page.tsx`
- `web/lib/types.ts`
- `web/lib/api.ts`
- `web/tests/api.test.ts`
- `.superpowers/sdd/2026-09-01-frontend-dashboard/task-1-report.md`

## Commit

Committed: `feat: add Next.js frontend foundation`.

## Concerns

- The local Node runtime is v23.7.0 while Vitest 4 declares support for Node 20, 22, or 24+. Tests nevertheless passed; a supported Node LTS version should be used in CI.
- `npm install` reports three high-severity transitive dependency advisories. No automated remediation was applied because it may alter the declared dependency graph.

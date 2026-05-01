---
title: "NotebookLM Playwright Automation — Operational Notes"
date: 2026-04-15
created_by: Claude Code (Claude Sonnet 4.6)
updated_by: Claude Code (Claude Sonnet 4.6)
updated_on: 2026-04-24
context: >
  NotebookLM Playwright automation debugging notes;
  Recurring issues encountered while automating NotebookLM source uploads
  via Playwright (headless Chrome), spinner/loading issues, indexing problems.
---

# NotebookLM Playwright Automation — Operational Notes

Debugging log for recurring issues encountered while automating NotebookLM
source uploads via Playwright (headless Chrome).  Read this before debugging
spinner or indexing problems.

---

## Problem: sources keep showing a spinning loader after upload

**Symptom**: sources appear in the NotebookLM panel, but the loading spinner
never goes away.  The source has no content and cannot be used.

**Root cause — hidden spinner, not removed**

When NotebookLM's backend finishes indexing a source it sets
`display: none` on the `mat-progress-spinner.loading-spinner` element
rather than removing it from the DOM.  Any code that counts spinner *elements*
(not just *visible* ones) will always see a non-zero count and incorrectly
report "still indexing" forever.

**Fix in `notebooklm_auth.py`**

`wait_all_indexed()` uses the `:visible` pseudo-class in its locator:

```python
spinner_visible = self.page.locator(
    "div.single-source-container mat-progress-spinner.loading-spinner:visible"
)
```

Playwright's `:visible` matches only elements that are rendered and occupy
space, so `count()` returns 0 once every spinner is hidden — the correct
"all done" signal.

**Cross-reference**: `delete_source_by_name()` uses
`wait_for(state="hidden")` on the same spinner locator before attempting
deletion.  "hidden" and "`:visible` count == 0" are consistent — both treat
the spinner hiding (not removal) as the completion signal.

---

## Problem: spinner / source count is 2× or 3× the actual source count

**Symptom**: `_snapshot_source_states()` reports `RAG-strategy.md` two or
three times in the "still indexing" list.  The count grows over time as
Angular builds its component tree.

**Root cause — Angular nests `div.single-source-container`**

NotebookLM's Angular component renders multiple nested
`div.single-source-container` wrappers around a single source card.  The
number of nesting levels increases progressively as Angular's change detection
runs, so a plain `querySelectorAll('div.single-source-container')` returns
2–3× the actual source count.

**Fix**: `wait_all_indexed()` counts spinner *elements*, not containers.
Each physical `mat-progress-spinner` exists once in the DOM regardless of
how many `div.single-source-container` wrappers surround it, so the element
count is always correct.

`_snapshot_source_states()` uses a `parentElement.closest()` filter to keep
only top-level containers, but this is fragile against Angular rendering
timing — prefer spinner-element counting for any "is still indexing?" check.

---

## Problem: Google session expires mid-session

**Symptom**: sync fails with "Google session expired" after a previous run
held the browser open for a long time (typically after a 10-minute
`wait_all_indexed()` timeout).

**Root cause**: Google session cookies are invalidated when a Playwright
browser session holds them open without interaction for an extended period,
or when NotebookLM receives conflicting session tokens from two concurrent
browser instances (login + headless sync).

**Fix**: Re-authenticate before retrying the sync:

```bash
DISPLAY=:1 python utils/notebooklm_auth.py --login
```

Complete the Google login in the browser window (3-minute timeout).
Then re-run the sync normally.  With the `:visible` spinner fix in place,
syncs should complete in 1–3 minutes (not 10) so session expiry is rare.

---

## Problem: a source is permanently stuck / duplicate sources appear

**Symptom**: a source has a permanent spinner with no content.  Or two
sources with the same name appear after a fresh upload.

**Root cause**: closing the browser (or the sync process exiting/timing out)
*while a source is actively indexing* leaves it in a broken state.
NotebookLM's backend stops processing but the spinner element stays in the
DOM indefinitely.  If the sync is then re-run without deleting the stuck
source first, Step 3's delete loop may only catch one of two copies.

**Fix**:

1. The Step 3 deletion loop (`for _ in range(5)`) re-queries after each
   deletion and repeats until no matching source is found — this correctly
   cleans up multiple stuck copies.

2. Always call `wait_all_indexed()` before closing the session (already done
   in Step 6 of `sync_to_notebook.py`).

3. If a source cannot be deleted programmatically (e.g. spinner never
   becomes hidden in the 2-minute deletion timeout), remove it manually in
   the NotebookLM UI: hover the source → **More (⋮)** → **Remove source**.

---

## Problem: all sources time out even though they indexed fine before

**Symptom**: `wait_all_indexed()` correctly reports genuine visible spinners
(`:visible` fix confirmed working) but all 3 sources remain at 3 visible
spinners for the full 10-minute timeout.  Even `codebase_snapshot.md`, which
normally indexes in under 60 seconds, times out.

**Root cause — NotebookLM backend throttling**

NotebookLM's backend throttles or stops processing entirely after too many
delete/upload cycles in a single session.  Confirmed 2026-04-17: after
~8 delete/upload cycles in one debugging session, the backend stopped
processing all new uploads regardless of file content or size.

This is NOT a code issue — the `:visible` spinner counting is correct and
accurately reflects the backend's refusal to process.

**Fix / prevention**:

1. **Run the sync only once per session** — at the very end of your coding
   session, not repeatedly during development.  Each debug re-run burns
   quota and risks triggering throttling.

2. **If throttled**: stop retrying.  Close the sync, leave the notebook
   as-is, and run `sync_to_notebook.py` the next day with a fresh
   `--login` session.  The backend resets between sessions.

3. **If you must re-run during a session**: use `--no-delete --no-snapshot`
   to avoid upload/delete cycles; only re-run with full delete+upload if
   sources are genuinely broken.

4. The "characters cause stuck indexing" hypothesis (box-drawing, arrows,
   dingbats) was investigated but NOT confirmed as the root cause — those
   characters are in `codebase_snapshot.md` too and it indexes fine.
   The sanitizer ranges added on 2026-04-17 may be premature; they are
   harmless but do not fix the throttling issue.

---

## What characters actually cause stuck indexing

Empirically confirmed ranges that cause the stuck-spinner / no-visible-error
state (collected 2026-04-17):

| Range | Examples | Action |
|-------|---------|--------|
| U+2500-U+257F | Box Drawing: ─ │ ┌ └ | Replace with `-` |
| U+2580-U+259F | Block Elements: █ ▓ ░ | Replace with `-` |
| U+2190-U+21FF | Arrows: → ← ↑ ↓ | Strip |
| U+2600-U+26FF | Misc Symbols: ☁ ⚠ ♠ | Strip |
| U+2700-U+27BF | Dingbats: ✅ ✓ ✗ | Strip |
| U+10000+ | Non-BMP emoji | Strip |

**Characters confirmed NOT to cause issues** (present in `codebase_snapshot.md`
which indexes reliably): `–` U+2013 (en dash), `—` U+2014 (em dash),
`≈` U+2248 (almost equal), `•` U+2022 (bullet), `§` U+00A7.

All stripping is handled by `_sanitize_for_markdown()` in
`utils/snapshot_codebase.py`.  Add new problem ranges there with a comment
and the date discovered.

---

## Problem: CDK overlay backdrop blocks upload button after fresh login

**Symptom**: `upload_source()` raises a Playwright timeout clicking
`button.add-source-button` even though the button is visible and enabled.
Log shows: "Dismissing open overlay before upload..." but the click still fails
because the backdrop never fully dismissed in the 5-second window.

**Root cause**: on a fresh Google session NotebookLM shows a "getting started"
modal (dark CDK backdrop) that can take more than one Escape press + a few
seconds to fully animate away. The original single-Escape + `wait_for(state=
"hidden", timeout=5_000)` was catching the timeout exception silently and
proceeding with the backdrop still blocking pointer events.

**Fix in `notebooklm_auth.py`**

`_dismiss_overlay()` uses JavaScript-based visibility checking (computed style,
not DOM presence) and retries up to 4 times:

```python
async def _dismiss_overlay(self) -> None:
    for attempt in range(4):
        visible = await self.page.evaluate(
            "Array.from(document.querySelectorAll('.cdk-overlay-backdrop'))"
            ".filter(el => { const s = window.getComputedStyle(el); "
            "return s.display !== 'none' && s.visibility !== 'hidden' "
            "&& parseFloat(s.opacity) > 0; }).length"
        )
        if visible == 0:
            break
        await self.page.keyboard.press("Escape")
        await self.page.wait_for_timeout(2_000)
```

`goto_notebook()` now calls `_dismiss_overlay()` right after the page loads
to pre-dismiss any welcome modal before any source operations begin.
Both `delete_source_by_name()` and `upload_source()` also call it.

**Confirmed working (2026-04-17)**: 2 Escape presses needed on a fresh session.

---

## Quick reference — selector cheat sheet

Confirmed working selectors as of 2026-04-17:

```
Source card container:     div.single-source-container
Source title:              [class*="source-title"]   (use .first per container)
Loading spinner:           mat-progress-spinner.loading-spinner
Visible spinner only:      mat-progress-spinner.loading-spinner:visible
More button:               button[aria-label="More"]
Upload button:             button.add-source-button
Upload files menu item:    button:has-text("Upload files")
Remove source menu item:   [role="menuitem"]:has-text("Remove source")
Delete confirm (v1):       button[aria-label="Confirm deletion"]
Delete confirm (v2):       .cdk-overlay-pane button:has-text("Delete")
Delete confirm (v3):       mat-dialog-container button:has-text("Delete")
```

NotebookLM updates its Angular component selectors occasionally.
`delete_source_by_name()` tries multiple delete-confirm selectors in order —
add new ones there if the UI changes.

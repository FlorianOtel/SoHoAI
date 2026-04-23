# NotebookLM Upload Issues — Findings & Workarounds

**Author**: Claude (claude-sonnet-4-6)
**Created**: 2026-04-15
**Updated**: 2026-04-15 (session 2 — encoding fix, deletion loop bugs; session 3 — split mode retired)
**Context**: End-of-session sync of HomeAI codebase to NotebookLM notebook
  `2f18268d-5e5b-4f40-8eaf-2909bbc945db`
**Files referenced**:
- `utils/snapshot_codebase.py`
- `utils/sync_to_notebook.py`
- `utils/notebooklm_auth.py`

> **Current state (session 3):** Split mode and the `.txt` workaround have been
> retired. `sync_to_notebook.py` uploads a single `codebase_snapshot.md` (all
> 15 files, ~122 KB) plus three separate doc Markdown files. The code paths below
> marked ~~strikethrough~~ are historical only — they no longer exist in the codebase.

---

## Problem 1 — Large combined snapshot (~116 KB) gets permanently stuck indexing

### Symptoms

After uploading `codebase_snapshot.md` (116.8 KB, 15 files), NotebookLM showed
a persistent `mat-progress-spinner.loading-spinner` on the source card for over
**25 minutes** without ever completing. The notebook itself appeared functional
(existing doc sources were fully indexed and available), but the new source never
left the "indexing" state. A page reload confirmed the spinner was not a stale
UI artifact — the backend genuinely never resolved.

No error state was surfaced in the DOM: `[class*="error"]` and `[class*="failed"]`
selectors returned 0 matches; only the spinner element was present.

### Root cause (corrected — see Problem 4 and Problem 5)

Initial hypothesis was file size. **This was wrong.** The true root cause was
Unicode encoding (Problem 4). Once encoding was fixed, a 122 KB combined snapshot
indexed in under 30 seconds (Problem 5).

### ~~Workaround — split into three files~~ (retired — size was a red herring)

> **Retired in session 3.** Split mode, `generate_split_snapshots()`,
> `codebase_snapshot_1/2/3.md`, and `nfs_files_mcp_server.txt` have all been
> removed from the codebase. Documented here for trailback only.

During session 1, before the true root cause was identified, the snapshot was split
into three parts to work around the assumed size limit:

| File | Contents | Size |
|------|----------|------|
| ~~`codebase_snapshot_1.md`~~ | core files | 54.6 KB |
| ~~`codebase_snapshot_2.md`~~ | RAG, utils | 42.3 KB |
| ~~`codebase_snapshot_3.md`~~ | NFS server | 23.0 KB |

Parts 1 and 2 indexed fine; Part 3 stuck — which pointed to content, not size.

---

## Problem 4 — Unicode box-drawing characters / emoji break NotebookLM's Markdown parser (root cause)

### Discovery

After the three-way split, Part 3 (only 23.0 KB — just `nfs_files_mcp_server.py`) also
stuck indefinitely, disproving the size hypothesis. Parts 1 (54.6 KB) and 2 (42.3 KB)
indexed fine. The only meaningful difference was the *content* of the file.

`nfs_files_mcp_server.py` contains Unicode characters used in its formatted output:

```python
lines = [f"📁 {root.name}/  ({root})\n"]           # U+1F4C1 FOLDER emoji
lines.append(f"{prefix}⚠️  [permission denied]")    # U+26A0 WARNING emoji
lines.append(f"{prefix}{connector}📄 {entry.name}") # U+1F4C4 PAGE emoji
numbered_lines.append(f"{i:4d} │ {line}")           # U+2502 BOX DRAWINGS LIGHT VERTICAL
header = f"...{'─' * 60}\n"                         # U+2500 BOX DRAWINGS LIGHT HORIZONTAL
```

Even inside a Markdown code fence, NotebookLM's parser chokes on these characters
and never completes indexing. There is no DOM-visible error — just a perpetual spinner.

### ~~Fix (session 1) — upload as raw `.txt`~~ (retired — see Problem 5)

> **Retired in session 3.** The `.txt` workaround and `nfs_files_mcp_server.txt`
> no longer exist. Documented here for trailback only.

Uploading `nfs_files_mcp_server.py` as `nfs_files_mcp_server.txt` (identical bytes,
just a `.txt` extension) indexed successfully in under 10 seconds. This confirmed
the parser was the issue, not the file's content per se.

### Fix (session 2, current) — sanitize at snapshot generation time

`_sanitize_for_markdown()` in `snapshot_codebase.py` strips the offending characters
before embedding any file in a Markdown code fence:

```python
def _sanitize_for_markdown(content: str) -> str:
    content = re.sub(r"[\u2500-\u257F]", "-", content)  # box-drawing → dash
    content = re.sub(r"[^\u0000-\uFFFF]", "", content)  # non-BMP emoji → removed
    return content
```

`nfs_files_mcp_server.py` is now embedded directly in the combined snapshot with no
special handling. The `.txt` file and split mode are both gone.

---

## Problem 2 — Stuck sources cannot be deleted via the normal UI flow

### Symptoms

`delete_source_by_name()` in `notebooklm_auth.py` checks for the
`mat-progress-spinner.loading-spinner` element and waits up to 2 minutes for it
to disappear before clicking "More → Remove source". When a source is
permanently stuck, this wait always times out and the method returns `False`.

Even after the timeout, the "More" (`button[aria-label="More"]`) button carries
Angular's `mat-mdc-button-disabled` class and the HTML `disabled` attribute,
which prevents all normal and `force=True` Playwright clicks from opening the
context menu.

### Workaround — JavaScript attribute patch before clicking

The disabled state is enforced only by Angular's component model; removing the
attribute via `page.evaluate()` is sufficient to allow the click to propagate:

```python
await page.evaluate("""
    document.querySelectorAll('button[aria-label="More"]').forEach(btn => {
        btn.removeAttribute('disabled');
        btn.classList.remove('mat-mdc-button-disabled', 'mdc-button--disabled');
    });
""")
await more_btn.click()
```

After the menu opens, "Remove source" is clickable normally. The confirmation
dialog uses `button:has-text("Delete")` rather than
`button[aria-label="Confirm deletion"]` in at least some NotebookLM versions —
both selectors should be tried.

**This fix has not been merged into `delete_source_by_name()` yet.** It is
currently only in the inline diagnostic scripts run during this session. If
stuck-source deletion is needed again, the JS patch above should be incorporated
into the method (guarded by a flag like `force: bool = False`).

---

## Problem 3 — NOTEBOOK_URL was pointing to the wrong notebook

`notebooklm_auth.py` had the URL hardcoded to a previous notebook:

```
# before
NOTEBOOK_URL = "https://notebooklm.google.com/notebook/e4c34b27-e10b-4b84-a482-31e259bcab5b"

# after (updated 2026-04-15)
NOTEBOOK_URL = "https://notebooklm.google.com/notebook/2f18268d-5e5b-4f40-8eaf-2909bbc945db"
```

When switching notebooks, update this constant before running any sync.

---

---

## Problem 5 — Unicode encoding fix allows single-file combined upload (confirmed 2026-04-15)

### Finding

After implementing `_sanitize_for_markdown()` in `snapshot_codebase.py`, a single
combined snapshot of **all 15 project files at 122 KB** indexed correctly in
NotebookLM in under 30 seconds. This disproves the ~55 KB size limit hypothesis from
Problem 1 — the only real limit was Unicode encoding.

### What was confirmed

The original stuck-indexing on the 116 KB combined snapshot (Problem 1) was caused
entirely by the Unicode characters in `nfs_files_mcp_server.py` (Problem 4), not
by file size. Once those characters are sanitized, files well above 55 KB index fine.

### Fix applied to `snapshot_codebase.py`

```python
def _sanitize_for_markdown(content: str) -> str:
    content = re.sub(r"[\u2500-\u257F]", "-", content)  # box-drawing → dash
    content = re.sub(r"[^\u0000-\uFFFF]", "", content)  # non-BMP emoji → removed
    return content
```

Applied in `generate_snapshot()` before wrapping each file in a code fence.

### Behaviour change (session 2 → further simplified in session 3)

- Single-file upload is the only mode. Split mode, `--split`, `--no-split`,
  `generate_split_snapshots()`, `DEFAULT_OUTPUT_PART1/2`, `SNAPSHOT_FILES_PART1/2`,
  and `NFS_SERVER_TXT` have all been removed from the codebase.
- `nfs_files_mcp_server.py` is embedded directly in `codebase_snapshot.md`
  via `_sanitize_for_markdown()`.
- `codebase_snapshot_1/2/3.md` and `nfs_files_mcp_server.txt` have been deleted
  from disk.

---

## Problem 6 — Confirmation dialog selector changed; silent deletion failure (confirmed 2026-04-15)

### Symptoms

`delete_source_by_name()` reported "Source deleted." for every source, but none
were actually removed.  `list_sources()` immediately after still returned the full
original list.  This caused the deletion while-loop to spin indefinitely, deleting
the same source name repeatedly with no effect.

### Root cause

`delete_source_by_name()` used `button[aria-label="Confirm deletion"]` to click the
confirmation dialog.  This selector no longer matches the current NotebookLM DOM.
The old code did:

```python
confirm = self.page.locator('button[aria-label="Confirm deletion"]')
if await confirm.count():   # ← always 0 now
    await confirm.click()
# ... returned True anyway — deletion never happened
```

Because the `if` branch was skipped silently, the method returned `True` without
ever confirming the dialog.  The source remained in the notebook, but the caller
treated it as deleted.

### Working selector (confirmed 2026-04-15)

```
.cdk-overlay-pane button:has-text("Delete")
```

### Fix applied to `delete_source_by_name()`

Now tries selectors in order, waits up to 3 s for each to become visible, and
returns `False` immediately if none match (instead of silently succeeding):

```python
for selector in [
    'button[aria-label="Confirm deletion"]',
    '.cdk-overlay-pane button:has-text("Delete")',
    'mat-dialog-container button:has-text("Delete")',
    'button.confirm-button',
]:
    try:
        btn = self.page.locator(selector).first
        await btn.wait_for(state="visible", timeout=3_000)
        await btn.click()
        confirmed = True
        break
    except Exception:
        continue

if not confirmed:
    await self.page.keyboard.press("Escape")
    return False
```

---

## Problem 7 — Deletion loop infinite due to DOM staleness (confirmed 2026-04-15)

### Symptoms

Even after a successful deletion (confirmation clicked, overlay dismissed), the
immediately-following `list_sources()` call still returned the deleted source.
The while-loop in `sync_to_notebook.py` therefore re-entered `delete_source_by_name()`
for the same source, producing an unbounded loop.

### Root cause

Angular's exit animation for the source card takes longer than the previous 1 s
`wait_for_timeout`.  `list_sources()` queries the live DOM and sees the card during
its exit transition, reporting the source as still present.

### Fix

Two-part:

1. **`notebooklm_auth.py`** — increased post-deletion wait from 1 s to 3 s to let
   Angular complete the DOM removal animation before returning from
   `delete_source_by_name()`.

2. **`sync_to_notebook.py`** — replaced the unbounded `while True:` with a
   bounded `for attempt in range(5):` loop to prevent infinite spinning even if
   DOM staleness persists beyond 3 s.

---

## Problem 8 — CDK overlay backdrop intercepts hover between successive deletions (confirmed 2026-04-15)

### Symptoms

When deleting multiple sources in sequence, the second call to
`delete_source_by_name()` failed with a Playwright `TimeoutError` because a
`cdk-overlay-backdrop` element was still visible and intercepting pointer events
on the target source container.

### Root cause

The `wait_for(state="hidden", timeout=8_000)` for the backdrop at the end of the
first deletion sometimes timed out or the backdrop reappeared before the second
hover attempt.

### Fix applied to `delete_source_by_name()`

Added an overlay-dismissal block at the **start** of each call:

```python
backdrop = self.page.locator(".cdk-overlay-backdrop")
if await backdrop.count() > 0:
    await self.page.keyboard.press("Escape")
    try:
        await backdrop.first.wait_for(state="hidden", timeout=5_000)
    except Exception:
        pass
    await self.page.wait_for_timeout(500)
```

This ensures any lingering backdrop from a previous operation is cleared before
attempting to hover on the next source container.

---

## Recommended future improvements

1. **Surface indexing errors** — NotebookLM may set additional CSS classes or
   data attributes when indexing fails silently. A future probe of the full
   `inner_html()` of a stuck container would help identify a reliable error
   selector.

2. **Detect selector drift proactively** — the confirmation dialog selector has
   already changed once. A probe that logs the full button text/attributes of
   anything inside `.cdk-overlay-pane` after clicking "Remove source" would make
   future drift easier to diagnose without a full debugging session.

3. **JS force-delete as last resort** — the JavaScript attribute-patch approach
   documented in Problem 2 (removing `mat-mdc-button-disabled` before clicking)
   could be added as a `force=True` fallback for stuck sources that never finish
   indexing.

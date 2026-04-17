"""
NotebookLM authentication and session management via Playwright.

First run  — opens a headed browser for manual Google OAuth login,
             then saves session cookies to NAS.
Subsequent — loads saved session, navigates directly to the notebook
             (headless, no login required).

Usage:
    # Authenticate (headed — opens browser window):
    python notebooklm_auth.py --login

    # Verify saved session still works:
    python notebooklm_auth.py --verify

    # Import as a module:
    from notebooklm_auth import NotebookLMSession
    async with NotebookLMSession() as session:
        await session.goto_notebook()
        # do things with session.page
"""

from __future__ import annotations

import asyncio
import argparse
import logging
import sys
from pathlib import Path

from playwright.async_api import async_playwright, BrowserContext, Page

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NOTEBOOK_URL = "https://notebooklm.google.com/notebook/2f18268d-5e5b-4f40-8eaf-2909bbc945db"
NOTEBOOKLM_HOME = "https://notebooklm.google.com"

# Session stored on NAS alongside the rest of the project
SESSION_FILE = Path("/mnt/nfs/Florian/Gin-AI/projects/HomeAI-Lab/utils/notebooklm_session.json")

# How long to wait for the user to complete Google login (seconds)
LOGIN_TIMEOUT_S = 180

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

class NotebookLMSession:
    """
    Async context manager that provides an authenticated Playwright page
    for the configured NotebookLM notebook.

    Usage:
        async with NotebookLMSession() as session:
            await session.goto_notebook()
            title = await session.page.title()
    """

    def __init__(
        self,
        headless: bool = True,
        session_file: Path = SESSION_FILE,
        notebook_url: str = NOTEBOOK_URL,
    ):
        self.headless = headless
        self.session_file = session_file
        self.notebook_url = notebook_url
        self._playwright = None
        self._browser = None
        self._context: BrowserContext | None = None
        self.page: Page | None = None

    async def __aenter__(self) -> "NotebookLMSession":
        self._playwright = await async_playwright().start()
        launch_kwargs = dict(
            channel="chrome",
            headless=self.headless,
            args=["--disable-blink-features=AutomationControlled", "--no-first-run"],
            ignore_default_args=["--enable-automation"],
        )
        ua = (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/145.0.0.0 Safari/537.36"
        )
        if self.session_file.exists():
            log.info("Loading saved session from %s", self.session_file)
            self._browser = await self._playwright.chromium.launch(**launch_kwargs)
            self._context = await self._browser.new_context(
                storage_state=str(self.session_file),
                user_agent=ua,
            )
        else:
            log.warning("No session file found — starting fresh (unauthenticated) context")
            self._browser = await self._playwright.chromium.launch(**launch_kwargs)
            self._context = await self._browser.new_context(user_agent=ua)

        await self._context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

        self.page = await self._context.new_page()
        return self

    async def __aexit__(self, *_):
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def goto_notebook(self, timeout_ms: int = 30_000) -> Page:
        """
        Navigate to the notebook URL and wait for it to load.

        Raises RuntimeError if the session is expired (Google redirects to
        accounts.google.com instead of opening the notebook).
        """
        await self.page.goto(self.notebook_url, wait_until="domcontentloaded", timeout=timeout_ms)
        # NotebookLM is Angular-rendered and never reaches networkidle, so we
        # wait for the basic load event and then give Angular a few seconds to
        # finish its initial render before any DOM queries.
        await self.page.wait_for_load_state("load", timeout=timeout_ms)
        await self.page.wait_for_timeout(4_000)

        # Session-expiry check: an expired session causes a redirect to
        # accounts.google.com before we ever see the notebook UI.  Detect this
        # early so the error surfaces here rather than as a confusing Playwright
        # timeout deep inside upload_source().
        url = self.page.url
        if "accounts.google.com" in url or "signin" in url.lower():
            raise RuntimeError(
                "Google session expired — re-authenticate with:\n"
                "  DISPLAY=:1 python utils/notebooklm_auth.py --login"
            )

        # Dismiss any "getting started" or welcome modal that appears on a fresh
        # session — if not dismissed here it will block upload_source() clicks.
        await self._dismiss_overlay()

        return self.page

    def is_authenticated(self) -> bool:
        """
        Return True if a session file exists on disk.

        NOTE: this only checks for file presence, not whether the session
        is still valid.  Use --verify (do_verify) or catch RuntimeError from
        goto_notebook() to detect an expired session at runtime.
        """
        return self.session_file.exists()

    # -- Overlay dismissal ----------------------------------------------------

    async def _dismiss_overlay(self) -> None:
        """
        Press Escape until any visible CDK overlay backdrop is gone.

        NotebookLM shows a dark-backdrop modal on fresh sessions ("getting
        started" panel) and occasionally leaves one open between operations.
        A single Escape + short wait is not always enough — this retries up
        to 4 times with JS-based visibility checking so we never proceed with
        a backdrop still blocking pointer events.
        """
        for attempt in range(4):
            visible = await self.page.evaluate(
                """() => Array.from(document.querySelectorAll('.cdk-overlay-backdrop'))
                   .filter(el => {
                       const s = window.getComputedStyle(el);
                       return s.display !== 'none' && s.visibility !== 'hidden'
                              && parseFloat(s.opacity) > 0;
                   }).length"""
            )
            if visible == 0:
                break
            log.info("Dismissing overlay (attempt %d, %d visible backdrop(s))...", attempt + 1, visible)
            await self.page.keyboard.press("Escape")
            await self.page.wait_for_timeout(2_000)

    # -- Source management ----------------------------------------------------

    async def list_sources(self) -> list[str]:
        """
        Return the display names of all sources currently in the notebook.

        Scopes the title query to each div.single-source-container so we get
        exactly one name per source.  A page-wide [class*='source-title'] query
        matches multiple nested elements per source (e.g. both a wrapper div
        and a child label whose class names both contain 'source-title'),
        producing duplicate entries for every source in the list.
        """
        containers = self.page.locator("div.single-source-container")
        count = await containers.count()
        names = []
        for i in range(count):
            # .first ensures we take only the top-most title element per container
            title_el = containers.nth(i).locator("[class*='source-title']").first
            txt = (await title_el.inner_text()).strip() if await title_el.count() else ""
            if txt:
                names.append(txt)
        return names

    async def delete_source_by_name(self, name: str) -> bool:
        """
        Delete a source whose display name contains `name` (case-insensitive).
        Returns True if deleted, False if not found.

        Flow (confirmed by DOM probe):
          container = div.single-source-container
          title     = [class*='source-title'] inside container
          more btn  = button[aria-label='More'] inside container
        """
        # Dismiss any lingering CDK overlay (modal, menu, dialog) that would
        # intercept pointer events on the target container — common when
        # called in rapid succession after a previous deletion.
        await self._dismiss_overlay()

        containers = self.page.locator("div.single-source-container")
        count = await containers.count()
        for i in range(count):
            container = containers.nth(i)
            title_el = container.locator("[class*='source-title']").first
            title = (await title_el.inner_text()).strip() if await title_el.count() else ""
            if name.lower() not in title.lower():
                continue

            log.info("Deleting source: %r", title)
            # NotebookLM disables the "More" button (mat-mdc-button-disabled /
            # disabled="true") while it is indexing the source — clicking it
            # or force-clicking it has no effect until indexing completes.
            # Wait up to 2 minutes for the spinner to disappear; if it hasn't
            # gone by then the source is stuck (backend error or very slow) and
            # we return False so the caller can decide how to proceed.
            loading = container.locator("mat-progress-spinner.loading-spinner")
            if await loading.count() > 0:
                log.info("Source is still indexing — waiting up to 2 min...")
                try:
                    await loading.wait_for(state="hidden", timeout=120_000)
                    await self.page.wait_for_timeout(500)
                except Exception:
                    log.warning(
                        "Source %r is still indexing after 2 min — skipping delete. "
                        "You may need to remove it manually in the NotebookLM UI.",
                        title,
                    )
                    return False

            # Scroll into view, then hover to expand the action icons
            # (the container uses icon-and-menu-container-collapsed when idle).
            await container.scroll_into_view_if_needed()
            await container.hover()
            more_btn = container.locator("button[aria-label='More']").first
            # Wait for Angular to mark the button as enabled after the hover
            # event; fall back to force=True if the attribute lags.
            try:
                await more_btn.wait_for(state="enabled", timeout=5_000)
            except Exception:
                await more_btn.click(force=True)
                await self.page.wait_for_timeout(500)
            else:
                await more_btn.click()
            await self.page.wait_for_timeout(500)

            # Menu item text is "delete\nRemove source" (icon + label)
            remove_item = self.page.locator('[role="menuitem"]', has_text="Remove source").first
            await remove_item.click()

            # Wait for the confirmation dialog to appear, then click the confirm button.
            # NotebookLM has used multiple selectors across versions; try them in order.
            confirmed = False
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
                    log.info("Deletion confirmed via selector: %s", selector)
                    break
                except Exception:
                    continue

            if not confirmed:
                log.warning(
                    "Confirmation dialog not found for %r — source was not deleted. "
                    "Check if the NotebookLM dialog selector has changed.",
                    title,
                )
                # Dismiss any open menu/overlay and bail out
                await self.page.keyboard.press("Escape")
                return False

            # Wait for all CDK overlays (modals, menus, dialogs) to fully dismiss
            try:
                await self.page.locator(".cdk-overlay-backdrop").wait_for(
                    state="hidden", timeout=8_000
                )
            except Exception:
                pass
            # Angular needs time to animate the source card out of the DOM.
            # 1 s was not enough — list_sources() immediately after still saw the
            # deleted source, causing the deletion loop to spin indefinitely.
            await self.page.wait_for_timeout(3_000)
            log.info("Source deleted.")
            return True

        log.warning("Source %r not found in notebook.", name)
        return False

    async def upload_source(self, file_path: Path, timeout_ms: int = 120_000) -> None:
        """
        Upload a file as a new source to the notebook.

        Flow (confirmed by DOM probe):
          1. Click "Add sources" button  (aria-label="Add source")
          2. Click "Upload files" from the menu
          3. Set files via the file chooser (no OS dialog needed)
          4. Wait for [class*='source-title']:has-text(stem) to appear

        NOTE: this method only confirms the source card appeared in the panel.
        The source will still be indexing (spinner visible) after this returns.
        Call wait_all_indexed() after all uploads to confirm indexing completes.
        """
        if not self.page:
            raise RuntimeError("Page not initialized — call goto_notebook() first.")

        log.info("Uploading source: %s", file_path)

        # Dismiss any open CDK overlay/dialog that would intercept pointer events
        # (e.g. the "getting started" modal on a fresh empty notebook)
        await self._dismiss_overlay()

        async with self.page.expect_file_chooser(timeout=10_000) as fc_info:
            # Use CSS class selector — more stable than aria-label after DOM mutations
            await self.page.locator("button.add-source-button").click()
            await self.page.wait_for_timeout(1_000)
            await self.page.locator("button", has_text="Upload files").first.click()

        fc = await fc_info.value
        await fc.set_files(str(file_path))
        log.info("File submitted — waiting for NotebookLM to process...")

        # Wait for the source title to appear in the panel (upload accepted).
        # The source will still be indexing at this point — see wait_all_indexed().
        stem = file_path.stem
        try:
            await self.page.locator(
                f"[class*='source-title']:has-text('{stem}')"
            ).first.wait_for(state="visible", timeout=timeout_ms)
            log.info("Source '%s' confirmed in panel (indexing may still be in progress).", stem)
        except Exception:
            log.warning("Could not confirm source by name — waiting 30s as fallback.")
            await self.page.wait_for_timeout(30_000)

    # -- Indexing monitor -----------------------------------------------------
    #
    # Problem: after upload_source() returns, NotebookLM spins a
    # mat-progress-spinner.loading-spinner on each card until its backend
    # finishes parsing and embedding the file.  If the browser session is
    # closed before indexing completes the source is left in a broken state
    # (perpetual spinner, no content) and must be manually removed + re-uploaded.
    #
    # Solution: call wait_all_indexed() after all uploads are submitted.
    # It polls every POLL_INTERVAL_MS and returns only when every source card
    # has its spinner gone and its "More" button enabled — or the timeout fires.
    # Failed sources (error indicator detected) are reported so the caller can
    # handle them (e.g. retry or warn the user).

    async def _snapshot_source_states(self) -> list[dict]:
        """
        Atomically read state for every div.single-source-container via a
        single JS evaluation.  Using JS avoids race conditions that arise when
        querying individual Playwright locators in a Python loop while Angular
        is mutating the DOM.

        Returns a list of dicts:
            title       — display name (or "(source-N)" if title not found)
            indexing    — True if a mat-progress-spinner.loading-spinner element
                          exists in the container (visible OR hidden).
                          NOTE: NotebookLM hides (display:none) the spinner when
                          indexing completes rather than removing it, so this flag
                          can be True for a fully-indexed source.  Use
                          wait_all_indexed() — which checks :visible spinners — as
                          the authoritative "still indexing" signal; this field is
                          only used here for error-state cross-checking.
            failed      — True if an error indicator is detected:
                            • any class containing "error" or "failed"
                            • mat-icon whose text content is "error"
                            • aria-label containing "error" or "failed" (case-insensitive)
                            • innerText containing "couldn't process" (NotebookLM backend error msg)
            more_disabled — True if the "More" button is disabled (usually means still indexing
                            or a failed state; cross-checked with the spinner flag above)
        """
        return await self.page.evaluate("""() => {
            const results = [];
            // querySelectorAll returns ALL matching elements including nested ones.
            // Angular sometimes wraps each card in an outer div that also carries
            // the same class, so we filter to only top-level containers — those
            // that are NOT themselves descendants of another single-source-container.
            const all = Array.from(document.querySelectorAll('div.single-source-container'));
            const containers = all.filter(
                c => !c.parentElement?.closest('div.single-source-container')
            );
            containers.forEach((c, i) => {
                const titleEl = c.querySelector('[class*="source-title"]');
                const title = titleEl?.innerText?.trim() || `(source-${i})`;

                const hasSpinner = !!c.querySelector('mat-progress-spinner.loading-spinner');

                // Error heuristics — NotebookLM has used several patterns across versions.
                const hasError = !!(
                    c.querySelector('[class*="error"]') ||
                    c.querySelector('[class*="failed"]') ||
                    Array.from(c.querySelectorAll('mat-icon')).some(
                        el => el.textContent?.trim().toLowerCase() === 'error'
                    ) ||
                    Array.from(c.querySelectorAll('[aria-label]')).some(
                        el => /error|failed/i.test(el.getAttribute('aria-label') || '')
                    ) ||
                    c.innerText?.toLowerCase().includes("couldn't process")
                );

                const moreBtn = c.querySelector('button[aria-label="More"]');
                const moreDisabled = !!(
                    moreBtn?.disabled ||
                    moreBtn?.classList.contains('mat-mdc-button-disabled')
                );

                results.push({ title, indexing: hasSpinner, failed: hasError,
                                more_disabled: moreDisabled });
            });
            return results;
        }""")

    async def wait_all_indexed(
        self,
        timeout_ms: int = 600_000,
        poll_interval_ms: int = 8_000,
    ) -> dict[str, str]:
        """
        Block until every source in the notebook finishes indexing, then return
        a summary dict mapping source name → final state string:

            "indexed"  — spinner gone, no error indicator detected
            "failed"   — error indicator detected (source needs manual review)
            "timeout"  — still spinning when the total timeout expired

        Parameters
        ----------
        timeout_ms : int
            Maximum total wait in milliseconds.  Default 600 000 (10 minutes).
            NotebookLM's backend can be slow on large files; 10 min is generous
            but avoids leaving the browser open indefinitely.
        poll_interval_ms : int
            How often to re-query the DOM.  Default 8 000 (8 seconds).
            Shorter intervals just add log noise without helping.

        Usage
        -----
            # After all upload_source() calls:
            states = await session.wait_all_indexed()
            if any(s != "indexed" for s in states.values()):
                print("WARNING: some sources did not index cleanly", states)

        Why this is necessary
        ---------------------
        Closing the browser before indexing completes leaves sources in a
        broken state (perpetual spinner, no content) that requires manual
        removal in the NotebookLM UI.  This is a recurring problem — always
        call wait_all_indexed() before exiting the session after uploads.

        Why spinner elements, not containers
        -------------------------------------
        NotebookLM's Angular component renders multiple nested
        div.single-source-container wrappers around a single source card.
        The number of nesting levels grows as Angular progressively builds
        the component tree, so counting containers double- or triple-counts
        each source in a way that changes over time.

        Counting spinner ELEMENTS with a descendant selector works correctly:
        each physical mat-progress-spinner element exists once in the DOM
        regardless of how many container wrappers surround it.  When the
        backend finishes processing a source the spinner element is removed,
        reducing the count by exactly 1.
        """
        import time

        # Count only VISIBLE spinner elements.
        #
        # When NotebookLM finishes indexing a source, its loading spinner may
        # become hidden (display:none) rather than being removed from the DOM.
        # Counting all spinner elements with .count() would incorrectly keep
        # reporting "still indexing" forever.  The :visible pseudo-class in
        # Playwright matches only elements that are rendered and occupying space,
        # so it returns 0 once every spinner is hidden — which is the correct
        # "all done" signal.
        #
        # Cross-reference: delete_source_by_name() uses wait_for(state="hidden")
        # on the same spinner locator to block until indexing is done before
        # attempting deletion — consistent with the spinner going hidden, not
        # being removed, on completion.
        spinner_visible = self.page.locator(
            "div.single-source-container mat-progress-spinner.loading-spinner:visible"
        )
        # For diagnostic logging: total spinner elements (visible + hidden)
        spinner_all = self.page.locator(
            "div.single-source-container mat-progress-spinner.loading-spinner"
        )

        deadline = time.monotonic() + timeout_ms / 1000
        log.info("Waiting for all sources to finish indexing (timeout %ds)...", timeout_ms // 1000)

        while True:
            n_visible = await spinner_visible.count()
            if n_visible == 0:
                n_total = await spinner_all.count()
                log.info(
                    "All spinners hidden/gone (%d total, 0 visible) — indexing complete.",
                    n_total,
                )
                break

            elapsed = int(timeout_ms / 1000 - (deadline - time.monotonic()))
            log.info(
                "  [%ds elapsed] %d visible spinner(s) — next check in %ds",
                elapsed, n_visible, poll_interval_ms // 1000,
            )

            if time.monotonic() >= deadline:
                log.warning("Indexing timeout — %d visible spinner(s) still active.", n_visible)
                break

            await self.page.wait_for_timeout(poll_interval_ms)

        # Use list_sources() for final source names — it is reliable (proven)
        # and unaffected by the container-nesting issue.
        sources = await self.list_sources()
        still_visible = await spinner_visible.count()
        timed_out = still_visible > 0

        # Error detection (best-effort): only run when spinners are gone so that
        # _snapshot_source_states() isn't confused by in-progress indexing state.
        failed_sources: set[str] = set()
        if not timed_out:
            raw = await self._snapshot_source_states()
            failed_sources = {s["title"] for s in raw if s["failed"]}

        states: dict[str, str] = {}
        for name in sources:
            if timed_out:
                states[name] = "timeout"
            elif name in failed_sources:
                states[name] = "failed"
            else:
                states[name] = "indexed"
        return states


# ---------------------------------------------------------------------------
# Login flow (headed — requires display)
# ---------------------------------------------------------------------------

async def do_login(session_file: Path = SESSION_FILE) -> None:
    """
    Open a headed browser, let the user complete Google OAuth,
    then save the session to disk.

    Uses the system Chrome (not Playwright's bundled Chromium) and strips
    the --enable-automation flag so Google allows login.
    """
    log.info("Starting headed browser for Google login...")
    log.info("Complete the login in the browser window. You have %ds.", LOGIN_TIMEOUT_S)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            channel="chrome",          # use system Google Chrome, not bundled Chromium
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-first-run",
                "--no-default-browser-check",
            ],
            ignore_default_args=["--enable-automation"],
        )
        context = await browser.new_context(
            # Mimic a real user agent
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/145.0.0.0 Safari/537.36"
            ),
        )
        # Remove the navigator.webdriver property that Google checks
        await context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        page = await context.new_page()

        await page.goto(NOTEBOOKLM_HOME)

        # Wait until we land on notebooklm.google.com (post-login redirect)
        try:
            await page.wait_for_url(
                "**/notebooklm.google.com/**",
                timeout=LOGIN_TIMEOUT_S * 1000,
            )
            log.info("Login detected — saving session...")
        except Exception:
            log.error("Timed out waiting for login. Aborting.")
            await browser.close()
            sys.exit(1)

        session_file.parent.mkdir(parents=True, exist_ok=True)
        await context.storage_state(path=str(session_file))
        log.info("Session saved to %s", session_file)

        await browser.close()


# ---------------------------------------------------------------------------
# Verify saved session
# ---------------------------------------------------------------------------

async def do_verify(session_file: Path = SESSION_FILE) -> None:
    """Load saved session and confirm the notebook is accessible."""
    if not session_file.exists():
        log.error("No session file at %s — run with --login first.", session_file)
        sys.exit(1)

    async with NotebookLMSession(headless=True, session_file=session_file) as session:
        log.info("Navigating to notebook...")
        try:
            await session.goto_notebook(timeout_ms=30_000)
        except Exception as e:
            log.error("Navigation failed: %s", e)
            sys.exit(1)

        url = session.page.url
        title = await session.page.title()
        log.info("Current URL  : %s", url)
        log.info("Page title   : %s", title)

        # Check we're not on the Google login page
        if "accounts.google.com" in url or "signin" in url.lower():
            log.error("Session is expired or invalid — re-run with --login.")
            sys.exit(1)

        log.info("Session is valid.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NotebookLM session management")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--login",  action="store_true", help="Open headed browser for Google login")
    group.add_argument("--verify", action="store_true", help="Verify saved session is still valid")
    parser.add_argument(
        "--session", default=str(SESSION_FILE),
        help=f"Path to session JSON file (default: {SESSION_FILE})",
    )
    args = parser.parse_args()

    session_file = Path(args.session)

    if args.login:
        asyncio.run(do_login(session_file))
    elif args.verify:
        asyncio.run(do_verify(session_file))


if __name__ == "__main__":
    main()

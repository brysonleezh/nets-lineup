# Portal Rookie Page Worklog — NCAA Bridge Rookie Projections (v1)

Plain-language log, appended after every step. Newest entries at the bottom.

---

## 2026-08-14 — Kickoff, preconditions

Checked all data sources the spec lists before starting - all 11 files
present on disk:

`data/projections/{nets_rookies_2026.csv,predictions_frozen.json}`,
`data/translator/{holdout_predictions.csv,holdout_metrics.csv,deployment/manifest.json,
deployment/posterior.npz,deployment_preprocessing.json}`,
`data/anchors/anchors.csv`, `data/college/recipes.csv`,
`reports/college_archetypes.md`, `data/basis_2025_26/archetype_labels.csv`.

Archetype labels precondition: `reports/college_archetypes.md` has zero
`(draft...)` markers remaining (finalized in Phase 6's own kickoff) - both
sides have real labels, no page will need to fall back to showing a bare
index.

Starting with a review of the current portal structure and an existing
page's conventions before writing any new code, per the spec's own
"match existing visual conventions... do not introduce a new charting
stack" instruction.

Reviewed `portal.py` (current line numbers for the import block, nav_options
list, sidebar-widget elif chain, and dispatch elif chain - all match what
an earlier same-session exploration found, unchanged), `portal_shared.py`
(the `BL_*` color tokens, the `@st.cache_data` loader convention, and
`_build_sortable_table_html` - a generic sortable-table-via-st.iframe
helper already used by every sortable table in the app, directly reusable
here), and found the app's own **existing validated 8-color categorical
palette** (`#2a78d6/#eb6834/#1baf7a/#eda100/#e87ba4/#008300/#4a3aa7/#e34948`
- blue/orange/aqua/gold/magenta/green/violet/red, already used elsewhere
for archetype-adjacent charts, run through this project's own dataviz-
skill contrast validator against `BL_PAPER`). Adopting this exact palette
for archetype-index color-coding throughout the new page rather than
inventing a second one, per the spec's "match existing conventions"
instruction.

**A real mathematical inconsistency found and resolved before writing any
Section 4 code**: the spec's counterfactual architecture describes
`softmax(B_mean · x)` using the posterior-*mean* coefficient matrix, but
`nets_rookies_2026.csv`'s frozen predictions were computed as
`E[softmax(Bx)]` (averaged over all 3000 posterior samples - the same
method used everywhere else in Phases 4-6). Tested directly on Mikel
Brown Jr.'s real data: `softmax(mean(B)x)` vs. `mean(softmax(Bx))` differ
by **up to 0.58 percentage points** per archetype dimension - a real
Jensen's-inequality gap (averaging before vs. after a nonlinear function
are different quantities), not a bug, and about 4 orders of magnitude
past the spec's own 1e-6 guard tolerance. Presented the finding plus two
resolution paths to the owner; **decision: export the full 3000-sample B
array and compute the counterfactual as `E[softmax(Bx)]`**, the same
method the frozen predictions use - guarantees exact (not approximate)
agreement, and remains fast (a vectorized matrix operation over 3000
already-stored samples, milliseconds, not new MCMC sampling - satisfies
the spec's own performance goal even though it exports slightly more than
the literal "posterior-mean B" text described).

**A second real, already-documented constraint found before writing
Section 3**: this exact codebase already tried and abandoned making a
`_build_sortable_table_html`-rendered table clickable
(`render_screening_table`'s own comment, `step3_player_breakdown.py`
~line 831) - `st.iframe()` sandboxes with a flag set that Streamlit gives
no public API to change, so `window.top.location` navigation from inside
the iframe throws a `SecurityError` in every browser, confirmed live in
an earlier session. Separately, `st.dataframe` (which supports native
row-selection) has ALSO already been ruled out for this app - it renders
via canvas (glide-data-grid) and reads the browser's own light/dark
toggle directly in JS, so its cell background comes out black regardless
of any CSS override (`portal.py`'s own CSS comment, ~line 115) - which is
exactly why every other table in this app uses the custom sortable-HTML-
iframe approach instead. Both of Section 3's two most obvious
implementations are each individually broken in this specific app for
reasons already discovered and documented elsewhere in the codebase.
**Resolution**: reuse the pattern this app already relies on for real
selection elsewhere (the Player Breakdown page's own sidebar
`st.selectbox`, a proven-working native widget) - the sortable HTML table
stays for browsing/sorting exactly as `_build_sortable_table_html` already
does everywhere else, and a separate `st.selectbox` (plus quick-select
buttons for the two spec-named notable cases) drives which player's
detail panel renders. Avoids both known failure modes rather than
re-discovering either one.

Built Sections 1-4 (`src/step5_rookie_projections.py`). While writing
Section 4, found a real, worth-fixing dependency problem shared between
this page and Phase 6's own script: `apply_frozen_transform`/
`build_rookie_raw_row`/`roster_min_start_by_athlete_id` (pure,
numpyro-independent functions this page's counterfactual needs) live in
`phase6_step2_predict_rookies.py`, which imported `numpyro`/`jax` at
**module level** for its own unrelated `predict_from_posterior` function -
meaning importing the pure functions would always drag in jax's own
import/JIT startup cost, directly working against this page's own <2s
cold-cache render requirement (and broke a standalone test outright under
plain `python3`, which has no numpyro installed). Fixed by making the
numpyro/jax imports lazy (moved inside `predict_from_posterior` and
`main()` specifically, including the `numpyro.set_host_device_count(4)`
call that must precede jax's own import) - confirmed the fix by importing
`apply_frozen_transform` under plain `python3` (no `.nets` venv, no
numpyro) and it worked cleanly.

**Ran the consistency guard for real** (not just written, executed):
`max_diff = 1.5e-8` across all 3 rookies at their real pick numbers -
comfortably inside the 1e-6 tolerance, confirming the pure-numpy
`E[softmax(Bx)]` path exactly reproduces the frozen jax-computed
predictions (expected: identical deterministic computation, numpy vs.
jax, no randomness involved since this reads already-stored posterior
samples rather than drawing new ones).

Wired the page into `portal.py` (4 edits: import, nav_options insert
immediately before "📄 Report" per spec, sidebar-widget elif branch - this
page owns its own selectbox/slider internally, so the branch is a no-op
placeholder rather than empty, matching the established pattern of every
page having an explicit branch - dispatch elif, module docstring
renumbering). Added the required unrelated-systems comment at
`SHOW_ROOKIE_SLOT_QUERY_PAGE`'s own definition in
`step3_player_breakdown.py`, not just at the new page's own entry point.

Wrote `tests/test_portal_rookie_page.py` (8 tests: AppTest default-load and
page-switch smoke tests, the consistency guard on real data AND a
deliberately-corrupted-posterior negative control - proof the guard
actually discriminates, not just always reports passed - the pure-numpy
prediction function's own simplex validity, graceful degradation on a
missing file, and a grep-based no-write-path guard).

**First AppTest run caught a real bug**: `st.iframe(..., scrolling=True)` -
this Streamlit version's `st.iframe()` has no `scrolling` kwarg at all
(`TypeError`, not a warning). Checked how every other `_build_sortable_table_html`
caller in this app invokes `st.iframe()` (grep across
`step2_intro.py`/`step3_player_breakdown.py`) - none pass `scrolling`, all
just use the table's own computed, uncapped `iframe_height`. Fixed by
removing the invalid kwarg AND removing the height cap I'd added (which,
without a working scroll mechanism, would have silently clipped most of a
36-row table) - matches the established pattern exactly rather than
inventing a new one. Re-ran: 8/8 passing.

## 2026-08-14 — Real-browser visual verification, three more bugs found

Per this project's CLAUDE.md UI-testing requirement ("start the dev server
and use the feature in a browser before reporting the task as complete"),
launched a real server (`streamlit run src/portal.py --server.address
127.0.0.1 ...` - bound explicitly, since an earlier session already found
the IPv6-wildcard default hangs this Streamlit version) and drove it with
Playwright, screenshotting every section.

**Bug 1 - "undefined" bars, no labels.** First screenshot of Sections 1-2
showed bar charts with no visible category labels, each reading
"undefined". Root cause: `load_labels()` returned `nba_labels` as the raw
`{index: {label, exemplar, match_quality, note}}` dict from
`load_nba_labels()` (built for the rookie-card prose, which needs the
extra fields), but `archetype_bar_chart()` and every other call site on
this page expected `{index: label_string}` - Plotly silently stringified
the dict's Python repr as "undefined" rather than erroring. Fixed by
flattening inside `load_labels()`: `nba_labels = {j: v["label"] for j, v
in nba_labels_full.items()}`.

**Bug 2 - a second, unrelated "undefined" above each chart.** After fixing
Bug 1, bars rendered correctly with real labels, but a stray "undefined"
text still appeared above every chart. Root cause: `archetype_bar_chart()`
called `fig.update_layout(title=title)` unconditionally, and most call
sites pass `title=None` - a confirmed-live Plotly.js quirk where setting
`title` to `None` explicitly renders the literal string "undefined"
instead of no title. Fixed by only calling `fig.update_layout(title=title)`
when `title` is truthy.

**Bug 3 - Section 3's College column showed "—" for all 36 rows, always.**
Caught by reading the screenshot literally rather than assuming "—" was
the intended empty-state for a few rows - every single row showed it.
Root cause: the code read `r.get("college_team", ...)` off `holdout_df`
(`holdout_predictions.csv`, a Phase 5 deliverable), but that file was
never built with a college-name column at all - `r.get()` on a pandas Series
falls back to the default silently rather than raising, so this shipped
looking like a working feature. Checked `anchors.csv` (already loaded
elsewhere on this same page for Section 4) and confirmed it has
`college_team` and covers all 36 holdout players by `player_name` with
zero misses (`anchors.csv` is 273 rows = the full train+holdout anchor
set, not just the 237 training anchors). Fixed by left-joining
`anchors_df[["player_name", "college_team"]]` onto the holdout frame
inside `render_section3_holdout_browser` (display-only - the function now
takes `anchors_df` as a fourth argument; neither `holdout_predictions.csv`
nor `anchors.csv` themselves are modified, consistent with the page's
read-only architecture and the no-write-path test guard).

All three fixes are display/wiring bugs in the new page's own code, not in
any upstream Phase 1-6 artifact - re-verified via a fresh screenshot after
each fix and via the existing consistency-guard test (unaffected, since
none of these three bugs touched the counterfactual math).

Re-ran `tests/` in full after all three fixes: **113/113 passing.**
Confirmed all 6 sections render cleanly via Playwright screenshots at
three scroll depths (top, mid, and per-section scroll-into-view) - no
remaining "undefined" artifacts, real archetype labels and colors
throughout, real college names in the holdout browser, frozen-record
footer (including its refreeze-log warning) rendering correctly.
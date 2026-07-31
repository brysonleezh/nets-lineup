"""
Builds a small, git-trackable copy of data/nets_synergy.db for deployment -
data/nets_synergy_deploy.db. step0_data.py's DB_PATH falls back to this file
whenever the full nets_synergy.db isn't present (see that file's own
comment), which is exactly the situation a from-scratch `git clone` deploy
(e.g. Streamlit Community Cloud - git is the only deploy channel, there's no
separate file-upload step) would otherwise leave with no database at all.

AI-ASSISTED (Claude Code, chat)
Prompt: "if I want to deploy this application on streamlit, how should I do
with data folder? Should I still add them in .gitignore file?" - then,
after being asked to verify rather than assume: "I don't [think] all files
in data folder are necessary, you could check it. And the portal doesn't
need original db file right?"
Used: traced every live call site (not just grepped for "lineups") across
all 5 main files plus the hidden Rookie Slot Query tab to confirm the live
app only ever queries `lineups` at GROUP_QUANTITY=2, SEASON='2025-26' -
every other season/group-size row (3/4/5-man lineups, and every season but
the current one) is real, but only used by the offline src/pipeline/
scripts (the abandoned synergy matrix, the RAPM investigation), never by
anything reachable from portal.py. Verified concretely, not estimated: a
real filtered copy measured 211MB -> 13MB. This script reproduces that
filter as a repeatable build step rather than a one-off manual copy.
Not AI: the decision to solve deployment this way (a filtered fallback
file + gitignore exception) rather than Git LFS or a remote-fetch-on-
startup pattern - the user's own call, made after seeing the size finding.

Run: python3 src/pipeline/build_deploy_db.py
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

# The live app never queries lineups outside this exact slice - see
# step0_data.py's own fallback comment for the full verification trail.
LIVE_GROUP_QUANTITY = 2
LIVE_SEASON = "2025-26"

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
REAL_DB_PATH = DATA_DIR / "nets_synergy.db"  # always the real file, never the deploy fallback - see step0_data.py
DEPLOY_DB_PATH = DATA_DIR / "nets_synergy_deploy.db"


def main():
    if not REAL_DB_PATH.exists():
        # Deliberately does NOT fall back to DEPLOY_DB_PATH the way
        # step0_data.py's DB_PATH does - regenerating the deploy db FROM
        # the deploy db would just re-filter an already-filtered file,
        # silently discarding the "real db is the source of truth"
        # invariant. Fail loudly instead of producing a misleading result.
        raise FileNotFoundError(
            f"Full nets_synergy.db not found at {REAL_DB_PATH} - this script "
            f"must run somewhere the real, complete database exists."
        )

    before_bytes = REAL_DB_PATH.stat().st_size
    print(f"source: {REAL_DB_PATH} ({before_bytes / 1e6:.1f} MB)")

    DEPLOY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DEPLOY_DB_PATH.exists():
        DEPLOY_DB_PATH.unlink()
    shutil.copy2(REAL_DB_PATH, DEPLOY_DB_PATH)

    conn = sqlite3.connect(DEPLOY_DB_PATH)
    try:
        n_before = conn.execute("SELECT COUNT(*) FROM lineups").fetchone()[0]
        conn.execute(
            "CREATE TABLE lineups_filtered AS SELECT * FROM lineups "
            "WHERE GROUP_QUANTITY = ? AND SEASON = ?",
            (LIVE_GROUP_QUANTITY, LIVE_SEASON),
        )
        conn.execute("DROP TABLE lineups")
        conn.execute("ALTER TABLE lineups_filtered RENAME TO lineups")
        n_after = conn.execute("SELECT COUNT(*) FROM lineups").fetchone()[0]
        conn.commit()
        conn.execute("VACUUM")
    finally:
        conn.close()

    after_bytes = DEPLOY_DB_PATH.stat().st_size
    print(f"lineups: {n_before} -> {n_after} rows "
          f"(kept GROUP_QUANTITY={LIVE_GROUP_QUANTITY}, SEASON={LIVE_SEASON!r})")
    print(f"deploy db: {DEPLOY_DB_PATH} ({after_bytes / 1e6:.1f} MB, "
          f"was {before_bytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

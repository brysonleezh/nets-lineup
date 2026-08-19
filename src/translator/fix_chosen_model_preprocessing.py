"""
Bug fix (Phase 5, caught in Step 1 before any holdout scoring happened):
chosen_model_preprocessing.json never saved the shot-type columns' own
post-imputation standardization mu/sd (only their imputation means) -
even though the actual fitted model's variant_b(df, df) call DID
standardize them (variant_b standardizes every continuous column,
shot-type included, after imputation). The deployed posterior samples
(chosen_model_posterior.npz) are untouched by this - this only corrects
an incomplete transcription of already-fitted, already-frozen statistics.

Recomputes the complete preprocessing dict from train_2017_2024.csv only
(never holdout_2025.csv) via the exact same variant_b(df, df) call Phase 4
used, so the corrected file is provably consistent with what was actually
fitted, not a new choice.

Run: .nets/bin/python src/translator/fix_chosen_model_preprocessing.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "translator"))

from input_variants import variant_b, Z_FEATURE_COLS, SHOT_TYPE_COLS  # noqa: E402

TRAIN_PATH = REPO_ROOT / "data" / "anchors" / "train_2017_2024.csv"
TRANSLATOR_DIR = REPO_ROOT / "data" / "translator"
RAW_COLS_FOR_STD = ["overall", "log_pick", "age_at_draft", "years_in_college", "PORPAG"]


def main() -> None:
    df = pd.read_csv(TRAIN_PATH)
    assert len(df) == 237

    X_full, X_check, col_names = variant_b(df, df)
    assert np.allclose(X_full, X_check)

    old = json.loads((TRANSLATOR_DIR / "chosen_model_preprocessing.json").read_text())

    # Recompute standardization mu/sd exactly as variant_b does internally: on the
    # continuous columns AFTER shot-type imputation (imputation itself uses the
    # frozen training-fold mean, already correctly saved in the old file).
    impute_means = old["shot_type_imputation_means"]
    shot_imputed = df[SHOT_TYPE_COLS].copy()
    for c in SHOT_TYPE_COLS:
        shot_imputed[c] = shot_imputed[c].fillna(impute_means[c])
    cont_cols = Z_FEATURE_COLS + RAW_COLS_FOR_STD + SHOT_TYPE_COLS
    cont_source = df[Z_FEATURE_COLS + RAW_COLS_FOR_STD].copy()
    cont_source[SHOT_TYPE_COLS] = shot_imputed
    mu_full = cont_source[cont_cols].mean()
    sd_full = cont_source[cont_cols].std()

    # verify this reproduces variant_b's real output exactly, for the non-boolean columns
    standardized_check = ((cont_source[cont_cols] - mu_full) / sd_full).values
    assert np.allclose(standardized_check, X_full[:, :len(cont_cols)], atol=1e-10), \
        "recomputed standardization doesn't match variant_b's actual output - do not trust this fix"

    old["standardization_mu"] = mu_full.to_dict()
    old["standardization_sd"] = sd_full.to_dict()
    old.pop("shot_type_standardization_mu_after_impute", None)  # dead placeholder field, remove
    old["note"] = (old["note"] + " [Corrected 2026-08-13: standardization_mu/sd now include the "
                   "shot-type columns' own post-imputation statistics, previously missing - a "
                   "documentation-completeness bug caught in Phase 5 Step 1, before any holdout "
                   "scoring. The deployed posterior samples in chosen_model_posterior.npz are "
                   "unaffected; this file's prior version simply never fully recorded what "
                   "variant_b(df, df) had already computed and used to fit that posterior.]")

    (TRANSLATOR_DIR / "chosen_model_preprocessing.json").write_text(json.dumps(old, indent=2, default=str))
    print(f"Fixed: standardization_mu/sd now have {len(old['standardization_mu'])} keys "
          f"(was {len(json.loads(Path(TRANSLATOR_DIR / 'chosen_model_preprocessing.json').read_text())['continuous_columns'])} continuous columns total, all now covered)")
    print("Verified: recomputed standardization exactly reproduces variant_b(df, df)'s real output (atol=1e-10)")


if __name__ == "__main__":
    main()

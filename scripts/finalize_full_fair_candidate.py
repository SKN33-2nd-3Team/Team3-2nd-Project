"""Finalize the saved candidate's import path and validate its raw-input schema.

No estimator is fitted here.  The script replaces only the stateless feature
transformer in the already-fitted pipeline, verifies prediction equivalence,
and records pass/fail behavior for expected app-input cases.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.full_fair_features import FullFairFeatureEngineer


RUN_ID = "20260721_full_fair_v1"
RUN = ROOT / "experiments" / "submission_full_comparison" / RUN_ID
CANDIDATE_DIR = ROOT / "models" / "candidates" / RUN_ID
CANDIDATE_PATH = CANDIDATE_DIR / "candidate_pipeline.joblib"
LEGACY_PATH = CANDIDATE_DIR / "candidate_pipeline_pre_importable_transformer.joblib"
METADATA_PATH = CANDIDATE_DIR / "metadata.json"
SMOKE_CSV = ROOT / "artifacts" / "app_schema_smoke_test.csv"
SMOKE_REPORT = ROOT / "reports" / "streamlit_candidate_compatibility.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def run_case(name: str, expected: str, operation) -> dict[str, str]:
    try:
        operation()
        outcome = "accepted"
        detail = "prediction completed"
    except Exception as exc:  # expected rejection cases are recorded explicitly
        outcome = "rejected"
        detail = f"{type(exc).__name__}: {str(exc)[:180]}"
    return {
        "case": name,
        "expected_behavior": expected,
        "actual_behavior": outcome,
        "status": "PASSED" if outcome == expected else "FAILED",
        "detail": detail,
    }


def main() -> None:
    from scripts.run_full_fair_comparison import Features

    # Preserve the original checkpoint exactly once for audit/recovery.
    if not LEGACY_PATH.exists():
        shutil.copy2(CANDIDATE_PATH, LEGACY_PATH)

    legacy = joblib.load(CANDIDATE_PATH)
    test = pd.read_csv(ROOT / "data" / "test.csv")
    sample = test.head(20).copy()
    before = legacy.predict_proba(sample)[:, 1]

    variant = legacy.named_steps["features"].variant
    legacy.set_params(features=FullFairFeatureEngineer(variant=variant))
    after = legacy.predict_proba(sample)[:, 1]
    if not np.allclose(before, after, rtol=0, atol=1e-12):
        raise RuntimeError("Replacing the stateless transformer changed predictions.")

    joblib.dump(legacy, CANDIDATE_PATH)
    reloaded = joblib.load(CANDIDATE_PATH)
    reload_probability = reloaded.predict_proba(sample)[:, 1]
    if not np.allclose(after, reload_probability, rtol=0, atol=1e-12):
        raise RuntimeError("Candidate predictions changed after reload.")

    fresh_process = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import joblib,pandas as pd; "
                "m=joblib.load(r'models/candidates/20260721_full_fair_v1/candidate_pipeline.joblib'); "
                "x=pd.read_csv(r'data/test.csv').head(2); "
                "p=m.predict_proba(x)[:,1]; "
                "assert len(p)==2 and ((p>=0)&(p<=1)).all(); print('FRESH_PROCESS_RELOAD_OK')"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if fresh_process.returncode != 0:
        raise RuntimeError(f"Fresh-process reload failed: {fresh_process.stderr}")

    base = test.head(5).copy()
    reordered = base.loc[:, list(reversed(base.columns))]
    unseen = base.copy(); unseen.loc[:, "subscription_type"] = "__UNSEEN_PLAN__"
    missing = base.drop(columns=["weekly_hours"])
    invalid = base.copy()
    invalid["weekly_hours"] = invalid["weekly_hours"].astype(object)
    invalid.loc[:, "weekly_hours"] = "not-a-number"
    extra = base.assign(app_only_note="ignored")

    cases = [
        run_case("normal_schema", "accepted", lambda: reloaded.predict_proba(base)),
        run_case("reordered_columns", "accepted", lambda: reloaded.predict_proba(reordered)),
        run_case("unseen_category", "accepted", lambda: reloaded.predict_proba(unseen)),
        run_case("extra_app_column", "accepted", lambda: reloaded.predict_proba(extra)),
        run_case("missing_required_numeric", "rejected", lambda: reloaded.predict_proba(missing)),
        run_case("invalid_numeric_type", "rejected", lambda: reloaded.predict_proba(invalid)),
    ]
    smoke = pd.DataFrame(cases)
    smoke.to_csv(SMOKE_CSV, index=False, encoding="utf-8-sig")
    if not smoke.status.eq("PASSED").all():
        raise RuntimeError("One or more schema smoke-test cases failed their expected behavior.")

    previous = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    feature_names = reloaded.named_steps["pre"].get_feature_names_out().tolist()
    metadata = {
        **previous,
        "status": "FULL_COMPARISON_COMPLETED_CANDIDATE_AWAITING_BUSINESS_REVIEW",
        "candidate_model": "catboost",
        "feature_variant": variant,
        "reload_verified": True,
        "fresh_process_reload_verified": True,
        "prediction_equivalence_after_transformer_relocation": True,
        "input_schema": {
            "identifier_column": "customer_id",
            "target_column": "churned",
            "input_columns": test.columns.tolist(),
            "model_required_columns": [column for column in test.columns if column != "customer_id"],
            "raw_dtypes": {column: str(dtype) for column, dtype in test.dtypes.items()},
            "column_reordering_supported": True,
            "unseen_categories_supported": True,
            "missing_required_columns_rejected": True,
            "invalid_numeric_types_rejected": True,
        },
        "transformed_feature_count": len(feature_names),
        "transformed_feature_names": feature_names,
        "package_versions": {
            "python": sys.version.split()[0],
            "pandas": version("pandas"),
            "numpy": version("numpy"),
            "scikit-learn": version("scikit-learn"),
            "catboost": version("catboost"),
            "joblib": version("joblib"),
        },
        "artifact_sha256": None,
        "schema_smoke_test": str(SMOKE_CSV.relative_to(ROOT)).replace("\\", "/"),
        "known_limitations": [
            "No untouched external labeled holdout is available for this run.",
            "No prediction horizon or churn event timestamp was supplied.",
            "OOF ranking evidence is not campaign uplift or causal effect evidence.",
            "The operating threshold requires a business decision about contact capacity and FN cost.",
        ],
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    # Hash after the final artifact has been written.  The hash is metadata
    # about the joblib and does not alter the joblib itself.
    metadata["artifact_sha256"] = sha256(CANDIDATE_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    columns = smoke.columns.tolist()
    markdown_rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in smoke.itertuples(index=False, name=None):
        markdown_rows.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    table = "\n".join(markdown_rows)
    SMOKE_REPORT.write_text(
        "# Full-fair candidate app-schema compatibility\n\n"
        "The saved CatBoost candidate loads in a fresh Python process and preserves predictions after moving the stateless feature transformer to an importable module. This completion step performed a schema compatibility test only; it did not replace the operational Streamlit artifact already promoted from the same run.\n\n"
        f"{table}\n\n"
        "- Reordered columns and unseen categories are accepted.\n"
        "- Missing required numeric fields and invalid numeric values are rejected instead of being silently coerced.\n"
        "- Candidate promotion, production approval, and the operating threshold remain user decisions.\n",
        encoding="utf-8",
    )

    manifest_path = RUN / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "COMPLETED_CANDIDATE_AWAITING_BUSINESS_REVIEW"
    manifest["blocked_reason"] = None
    manifest["checkpoints"]["setup"] = "PASSED_AFTER_RESOLUTION"
    manifest["checkpoints"]["app_schema_smoke_test"] = "PASSED"
    manifest["checkpoints"]["presentation_evidence"] = "PASSED"
    manifest["checkpoints"]["instructor_audit"] = "PASSED_WITH_EXTERNAL_HOLDOUT_LIMITATION"
    manifest["last_checkpoint"] = "final_audit"
    manifest["candidate_model"] = "catboost"
    manifest["external_labeled_holdout"] = "NOT_AVAILABLE"
    manifest["retraining_performed_during_completion"] = False
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"fresh_process_reload": True, "schema_cases": len(smoke), "all_cases_passed": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()

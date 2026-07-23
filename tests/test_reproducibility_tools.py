from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from tools.submission_hashing import canonical_file_info


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_requires_declared_outputs(tmp_path, monkeypatch) -> None:
    script = load_script("run_full_fair_comparison_test", ROOT / "scripts/run_full_fair_comparison.py")
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(
        json.dumps({"checkpoints": {"top3_fine_tuning": "PASSED"}}),
        encoding="utf-8",
    )
    required = run / "oof_fine_catboost.npy"
    monkeypatch.setattr(script, "RUN", run)
    monkeypatch.setattr(script, "required_outputs", lambda step: [required])

    assert not script.done("top3_fine_tuning")
    required.write_bytes(b"generated")
    assert script.done("top3_fine_tuning")


def test_fresh_run_archives_generated_state(tmp_path, monkeypatch) -> None:
    script = load_script("run_full_fair_fresh_test", ROOT / "scripts/run_full_fair_comparison.py")
    root = tmp_path / "repo"
    run = root / "experiments" / "submission_full_comparison" / script.RID
    run.mkdir(parents=True)
    (run / "run_config.json").write_text("{}\n", encoding="utf-8")
    (run / "execution_plan.md").write_text("plan\n", encoding="utf-8")
    (run / "run_manifest.json").write_text('{"checkpoints":{"top3_fine_tuning":"PASSED"}}\n', encoding="utf-8")
    (run / "oof_fine_catboost.npy").write_bytes(b"old")
    monkeypatch.setattr(script, "ROOT", root)
    monkeypatch.setattr(script, "RUN", run)

    archive = script.prepare_fresh_run()

    assert (archive / "oof_fine_catboost.npy").read_bytes() == b"old"
    assert (archive / "run_manifest.json").exists()
    assert (run / "run_config.json").exists()
    assert not (run / "oof_fine_catboost.npy").exists()
    fresh = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    assert fresh["status"] == "IN_PROGRESS"
    assert fresh["checkpoints"] == {"repository_and_data_audit": "PASSED"}


def test_submission_text_hash_is_eol_independent(tmp_path) -> None:
    lf = tmp_path / "lf.csv"
    crlf = tmp_path / "crlf.csv"
    lf.write_bytes(b"a,b\n1,2\n")
    crlf.write_bytes(b"a,b\r\n1,2\r\n")

    assert canonical_file_info(lf) == canonical_file_info(crlf)


def test_submission_allowlist_excludes_reproducible_intermediates() -> None:
    script = load_script(
        "build_submission_manifest_test",
        ROOT / "tools/build_submission_manifest.py",
    )

    assert script.included(ROOT / ".gitattributes")
    assert script.included(ROOT / "docs" / "submission_checklist.md")
    assert script.included(ROOT / "data" / "train.csv")
    assert not script.included(
        ROOT
        / "experiments"
        / "submission_full_comparison"
        / "20260721_full_fair_v1"
        / "oof_fine_catboost.npy"
    )
    assert not script.included(ROOT / "data" / "temporary_export.csv")


def test_promotion_bundle_stages_model_aliases_and_metadata(tmp_path) -> None:
    script = load_script("promote_full_fair_test", ROOT / "scripts/promote_full_fair_candidate.py")
    source = tmp_path / "candidate.joblib"
    source.write_bytes(b"candidate-model")
    app_model = tmp_path / "app" / "model.joblib"
    submission_model = tmp_path / "submission" / "model.joblib"
    metadata = tmp_path / "artifacts" / "metadata.json"
    metrics = tmp_path / "artifacts" / "metrics.csv"

    script.publish_bundle(
        copies={app_model: source, submission_model: source},
        texts={metadata: '{"artifact_sha256":"test"}\n'},
        frames={metrics: pd.DataFrame([{"model": "catboost", "pr_auc": 0.9}])},
    )

    assert app_model.read_bytes() == source.read_bytes()
    assert submission_model.read_bytes() == source.read_bytes()
    assert json.loads(metadata.read_text(encoding="utf-8"))["artifact_sha256"] == "test"
    assert pd.read_csv(metrics).loc[0, "model"] == "catboost"

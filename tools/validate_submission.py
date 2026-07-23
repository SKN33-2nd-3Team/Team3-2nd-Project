"""Validate the assembled PlaylistPro submission package."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import nbformat
import pandas as pd
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.submission_hashing import canonical_file_info

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    required = [
        "README.md", "docs/data_source.md", "reports/preprocessing_report.md",
        "reports/preprocessing_report.pdf", "reports/training_report.md",
        "reports/training_report.pdf", "notebooks/01_data_check.ipynb",
        "notebooks/02_eda.ipynb", "notebooks/03_model_experiments.ipynb",
        "models/churn_pipeline.joblib", "artifacts/feature_schema.json",
        "artifacts/model_metadata.json", "artifacts/metrics.csv",
        "app/streamlit_app.py", "submission_manifest.json",
    ]
    missing = [item for item in required if not (ROOT / item).exists()]
    assert not missing, f"missing files: {missing}"

    train = pd.read_csv(ROOT / "data/train.csv")
    test = pd.read_csv(ROOT / "data/test.csv")
    assert train.shape == (125_000, 20)
    assert test.shape == (75_000, 19)
    assert train.customer_id.is_unique and test.customer_id.is_unique
    assert not (set(train.customer_id) & set(test.customer_id))

    metadata = json.loads((ROOT / "artifacts/model_metadata.json").read_text(encoding="utf-8"))
    model_path = ROOT / "models/churn_pipeline.joblib"
    assert sha256(model_path) == metadata["artifact_sha256"]
    model = joblib.load(model_path)
    probability = float(model.predict_proba(test.head(1))[0, 1])
    assert 0 <= probability <= 1

    notebook_results = {}
    for name in ["01_data_check.ipynb", "02_eda.ipynb", "03_model_experiments.ipynb"]:
        notebook = nbformat.read(ROOT / "notebooks" / name, as_version=4)
        errors = [
            output
            for cell in notebook.cells if cell.cell_type == "code"
            for output in cell.get("outputs", []) if output.get("output_type") == "error"
        ]
        assert not errors, f"notebook errors: {name}"
        notebook_results[name] = {"cells": len(notebook.cells), "errors": len(errors)}

    pdf_pages = {}
    for name in ["preprocessing_report.pdf", "training_report.pdf"]:
        reader = PdfReader(ROOT / "reports" / name)
        assert len(reader.pages) >= 3
        pdf_pages[name] = len(reader.pages)
    manifest = json.loads((ROOT / "submission_manifest.json").read_text(encoding="utf-8"))
    manifest_paths = {row["path"] for row in manifest["files"]}
    required_manifest_paths = set(required) - {"submission_manifest.json"}
    assert required_manifest_paths <= manifest_paths, "required files are missing from submission manifest"
    forbidden = [
        path for path in manifest_paths
        if "__pycache__" in path or ".pytest_cache" in path
        or path.endswith((".pyc", ".pyo", ".inspect.ndjson"))
        or "prompt" in path.lower()
    ]
    assert not forbidden, f"forbidden files in submission manifest: {forbidden}"
    for row in manifest["files"]:
        file_path = ROOT / row["path"]
        size, digest = canonical_file_info(file_path)
        assert size == row["bytes"]
        assert digest == row["sha256"]

    result = {
        "status": "PASSED",
        "data": {"train": list(train.shape), "test": list(test.shape)},
        "model": {"sha256": metadata["artifact_sha256"], "sample_probability": probability},
        "notebooks": notebook_results,
        "pdf_pages": pdf_pages,
        "manifest_files": len(manifest["files"]),
    }
    (ROOT / "reports/submission_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

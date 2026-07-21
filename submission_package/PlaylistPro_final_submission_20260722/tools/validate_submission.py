"""Validate the assembled PlaylistPro submission package."""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import joblib
import nbformat
import pandas as pd
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    required = [
        "README.md", "DATA_SOURCE.md", "reports/preprocessing_report.md",
        "reports/preprocessing_report.pdf", "reports/training_report.md",
        "reports/training_report.pdf", "notebooks/01_data_check.ipynb",
        "notebooks/02_eda.ipynb", "notebooks/03_model_experiments.ipynb",
        "models/churn_pipeline.joblib", "artifacts/feature_schema.json",
        "artifacts/model_metadata.json", "artifacts/metrics.csv",
        "app/streamlit_app.py", "presentation/PlaylistPro_final_presentation.pptx",
        "presentation/PlaylistPro_final_presentation.pdf", "submission_manifest.json",
    ]
    missing = [item for item in required if not (ROOT / item).exists()]
    assert not missing, f"missing files: {missing}"

    train = pd.read_csv(ROOT / "data/processed/train.csv")
    test = pd.read_csv(ROOT / "data/processed/test.csv")
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
    deck_pdf = PdfReader(ROOT / "presentation/PlaylistPro_final_presentation.pdf")
    assert len(deck_pdf.pages) >= 10
    pdf_pages["PlaylistPro_final_presentation.pdf"] = len(deck_pdf.pages)

    with zipfile.ZipFile(ROOT / "presentation/PlaylistPro_final_presentation.pptx") as deck:
        slide_count = len([name for name in deck.namelist() if name.startswith("ppt/slides/slide") and name.endswith(".xml")])
    assert slide_count >= 10

    manifest = json.loads((ROOT / "submission_manifest.json").read_text(encoding="utf-8"))
    manifest_paths = {row["path"] for row in manifest["files"]}
    actual_paths = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path.name != "submission_manifest.json"
    }
    assert manifest_paths == actual_paths, "submission manifest does not match package contents"
    for row in manifest["files"]:
        file_path = ROOT / row["path"]
        assert file_path.stat().st_size == row["bytes"]
        assert sha256(file_path) == row["sha256"]

    result = {
        "status": "PASSED",
        "data": {"train": list(train.shape), "test": list(test.shape)},
        "model": {"sha256": metadata["artifact_sha256"], "sample_probability": probability},
        "notebooks": notebook_results,
        "pdf_pages": pdf_pages,
        "presentation_slides": slide_count,
        "manifest_files": len(manifest["files"]),
    }
    (ROOT / "reports/submission_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

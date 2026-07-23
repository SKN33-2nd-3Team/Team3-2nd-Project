"""Smoke-test all six Streamlit views with Streamlit's AppTest runner."""

from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
PAGES = [
    "프로젝트 요약",
    "데이터·고객 인사이트",
    "개선 실험",
    "모델 선정",
    "운영 시나리오",
    "고객 우선순위",
]


def main() -> None:
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=30)
    app.run()
    results: dict[str, dict[str, object]] = {}
    for page in PAGES:
        app.sidebar.radio[0].set_value(page)
        app.run()
        exceptions = [str(item.value) for item in app.exception]
        results[page] = {"status": "PASSED" if not exceptions else "FAILED", "exceptions": exceptions}
        if exceptions:
            raise AssertionError(f"{page}: {exceptions}")

    output = {"status": "PASSED", "pages": results}
    (ROOT / "reports/streamlit_validation.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

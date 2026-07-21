# Full-fair candidate app-schema compatibility

The saved CatBoost candidate loads in a fresh Python process and preserves predictions after moving the stateless feature transformer to an importable module. This completion step performed a schema compatibility test only; it did not replace the operational Streamlit artifact already promoted from the same run.

| case | expected_behavior | actual_behavior | status | detail |
| --- | --- | --- | --- | --- |
| normal_schema | accepted | accepted | PASSED | prediction completed |
| reordered_columns | accepted | accepted | PASSED | prediction completed |
| unseen_category | accepted | accepted | PASSED | prediction completed |
| extra_app_column | accepted | accepted | PASSED | prediction completed |
| missing_required_numeric | rejected | rejected | PASSED | KeyError: 'weekly_hours' |
| invalid_numeric_type | rejected | rejected | PASSED | TypeError: unsupported operand type(s) for /: 'str' and 'float' |

- Reordered columns and unseen categories are accepted.
- Missing required numeric fields and invalid numeric values are rejected instead of being silently coerced.
- Candidate promotion, production approval, and the operating threshold remain user decisions.

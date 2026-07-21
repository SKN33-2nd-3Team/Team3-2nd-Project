# 인사이트 기반 전처리 검토

| 전처리안 | Logistic OOF PR-AUC | 결정 | 근거 |
| --- | ---: | --- | --- |
| raw | 0.899495 | 제외 | log_numeric보다 낮음 |
| plus1_ratios | 0.899486 | 제외 | 비율 추가 효과가 확인되지 않음 |
| zero_aware_ratios | 0.899485 | 제외 | 0 분모 별도 처리의 추가 효과가 확인되지 않음 |
| log_numeric | **0.906293** | 채택 | Logistic OOF PR-AUC 최고 |
| interaction | 0.899637 | 제외 | 범주 상호작용의 추가 효과가 작음 |
| signal_pruned | 0.899493 | 제외 | 신호 축소가 성능을 높이지 못함 |

모든 전처리는 예측 시점의 원천 입력만 사용합니다. Target 기반 인코딩은 사용하지 않았습니다. 비율 Feature는 +1 완화와 0 분모 처리안을 비교했고, 치우친 행동 수치의 Log 변환을 포함한 `log_numeric`을 공통 Feature로 채택했습니다.

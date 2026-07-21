import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Presentation, PresentationFile, layers, shape, text } from "@oai/artifact-tool";

const ROOT = process.env.PLAYLISTPRO_SUBMISSION_ROOT
  || path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUT = path.join(ROOT, "presentation", "PlaylistPro_final_presentation.pptx");
const W = 1280;
const H = 720;
const FONT = "Malgun Gothic";
const C = {
  ink: "#0B1220", muted: "#526071", blue: "#2563EB", cyan: "#22D3EE",
  pale: "#EAF2FF", bg: "#F7F9FC", white: "#FFFFFF", line: "#D8E0EA",
  green: "#0F9F6E", amber: "#D97706", red: "#DC2626", navy: "#15233D",
};

function tx(content, left, top, width, height, fontSize = 24, color = C.ink, extra = {}) {
  return text([content], {
    position: { left, top }, width, height,
    style: {
      fontSize: `${fontSize}px`, typeface: FONT, color,
      alignment: extra.alignment ?? "left",
      verticalAlignment: extra.verticalAlignment ?? "top",
      autoFit: extra.autoFit ?? "shrinkText", wrap: "square",
      bold: extra.bold ?? false,
      insets: { top: 0, right: 0, bottom: 0, left: 0 },
    },
  });
}

function box(left, top, width, height, fill = C.white, radius = true, line = C.line) {
  return shape({
    geometry: radius ? "roundRect" : "rect", fill,
    line: { style: "solid", width: 1, fill: line },
    position: { left, top }, width, height,
  });
}

function pill(label, left, top, width, fill = C.pale, color = C.blue) {
  return [
    box(left, top, width, 30, fill, true, fill),
    tx(label, left + 10, top + 5, width - 20, 20, 14, color, { bold: true, alignment: "center" }),
  ];
}

function header(title, kicker, page) {
  return [
    tx(kicker.toUpperCase(), 48, 30, 420, 24, 14, C.blue, { bold: true }),
    tx(title, 48, 62, 1150, 62, 45, C.ink, { bold: true }),
    shape({ geometry: "rect", fill: C.blue, position: { left: 48, top: 136 }, width: 64, height: 5 }),
    tx(String(page).padStart(2, "0"), 1190, 674, 42, 20, 14, C.muted, { alignment: "right" }),
    tx("PlaylistPro · 최종 제출", 48, 674, 260, 20, 14, C.muted),
  ];
}

function makeSlide(pres, nodes, name) {
  const slide = pres.slides.add();
  slide.compose(
    layers({ name, width: "fill", height: "fill" }, [
      shape({ geometry: "rect", fill: C.bg, position: { left: 0, top: 0 }, width: W, height: H }),
      ...nodes,
    ]),
    { frame: { left: 0, top: 0, width: W, height: H }, baseUnit: 1 },
  );
  return slide;
}

async function addImage(slide, rel, left, top, width, height, fit = "contain", alt = "") {
  const full = path.join(ROOT, rel);
  const bytes = await fs.readFile(full);
  const dataUrl = `data:image/png;base64,${bytes.toString("base64")}`;
  const image = slide.images.add({ dataUrl, fit, alt: alt || path.basename(rel) });
  image.position = { left, top, width, height };
}

function metricCard(value, label, left, top, width, accent = C.blue) {
  return [
    box(left, top, width, 118, C.white, true),
    shape({ geometry: "rect", fill: accent, position: { left, top }, width: 7, height: 118 }),
    tx(value, left + 22, top + 19, width - 40, 42, 34, C.ink, { bold: true }),
    tx(label, left + 22, top + 70, width - 40, 25, 17, C.muted),
  ];
}

const p = Presentation.create({ slideSize: { width: W, height: H } });

// 1. Cover
let s = makeSlide(p, [
  shape({ geometry: "rect", fill: C.navy, position: { left: 0, top: 0 }, width: 1280, height: 720 }),
  ...pill("CUSTOMER RETENTION", 58, 54, 196, "#203456", "#7DD3FC"),
  tx("PlaylistPro", 58, 134, 560, 84, 66, C.white, { bold: true }),
  tx("누구에게 어떤 유지 활동을\n먼저 검토할 것인가?", 58, 235, 555, 142, 42, C.white, { bold: true }),
  tx("EDA → 공정 비교 → CatBoost 위험 순위 → 운영 선택지", 58, 410, 560, 58, 24, "#B8C8E3"),
  tx("최종 제출 패키지 · 2026.07.22", 58, 640, 420, 24, 17, "#8FA5C8"),
  box(665, 54, 565, 586, "#F8FAFC", true, "#33486A"),
], "playlistpro-cover");
await addImage(s, "assets/screenshots/01_project_summary.png", 682, 72, 531, 550, "contain", "PlaylistPro 프로젝트 요약 화면");

// 2. Decision flow
s = makeSlide(p, [
  ...header("예측 점수가 아니라 운영 의사결정까지 연결", "Problem → Decision", 2),
  tx("목표", 48, 168, 100, 26, 18, C.muted, { bold: true }),
  tx("이탈 가능성이 높은 고객을 찾고, 제한된 접촉 자원을 우선순위에 맞춰 검토한다.", 132, 165, 1035, 36, 25, C.ink, { bold: true }),
  ...[0,1,2,3].flatMap((i) => {
    const left = 48 + i * 296;
    const titles = ["1. 신호 확인", "2. 위험 순위화", "3. 운영안 선택", "4. 사후 검증"];
    const bodies = [
      "사용량·요금제·문의·일시정지 등 관찰 신호를 점검",
      "동일 조건 비교 후 CatBoost OOF 확률로 우선순위 생성",
      "Recall·Precision·접촉량·비용 가정을 함께 제시",
      "실제 캠페인 A/B 테스트와 미래 라벨로 효과 확인",
    ];
    const accents = [C.cyan, C.blue, C.amber, C.green];
    return [
      box(left, 245, 264, 260, C.white, true),
      shape({ geometry: "rect", fill: accents[i], position: { left, top: 245 }, width: 264, height: 8 }),
      tx(titles[i], left + 20, 280, 224, 40, 25, C.ink, { bold: true }),
      tx(bodies[i], left + 20, 345, 224, 115, 20, C.muted),
    ];
  }),
  box(48, 542, 1152, 78, "#FFF7E8", true, "#F3D6A3"),
  tx("책임 경계", 70, 564, 120, 28, 19, C.amber, { bold: true }),
  tx("앱은 행동 후보와 가정 기반 계획을 제시한다. 실제 고객 접촉·할인 집행·법적 승인은 조직이 결정한다.", 188, 559, 970, 36, 21, C.ink),
], "decision-flow");

// 3. Data
s = makeSlide(p, [
  ...header("규모는 충분하지만, 현실 일반화에는 명확한 한계", "Data & Scope", 3),
  ...metricCard("125,000", "학습 고객", 48, 176, 260, C.blue),
  ...metricCard("75,000", "미라벨 고객", 330, 176, 260, C.cyan),
  ...metricCard("51.34%", "학습 데이터 이탈률", 612, 176, 260, C.amber),
  ...metricCard("20개", "컬럼 (타깃 포함)", 894, 176, 306, C.green),
  box(48, 330, 550, 240, C.white, true),
  tx("확인된 데이터 품질", 72, 356, 470, 35, 26, C.ink, { bold: true }),
  tx("• 결측치 0\n• 중복 고객 0\n• Train/Test 키 중복 0\n• 범주·수치 스키마 검증 완료", 72, 412, 470, 130, 22, C.muted),
  box(630, 330, 570, 240, "#FFF3F2", true, "#F2C7C3"),
  tx("발표에서 반드시 밝힐 한계", 654, 356, 500, 35, 26, C.red, { bold: true }),
  tx("• 규칙 기반 합성 데이터로 판단\n• 예측 시점·이탈 horizon 미정의\n• 외부 라벨 Holdout 없음\n• 라이선스·Rules 최종 승인은 사람 확인 필요", 654, 412, 500, 136, 22, C.ink),
], "data-scope");

// 4. EDA to preprocessing
s = makeSlide(p, [
  ...header("EDA에서 발견한 신호를 최소한의 전처리로 연결", "EDA → Preprocessing", 4),
  box(48, 166, 430, 456, C.white, true),
  tx("관찰된 핵심 신호", 72, 192, 360, 36, 27, C.ink, { bold: true }),
  tx("01  weekly_hours 감소\n02  Free 요금제\n03  service inquiry 증가\n04  subscription pauses\n05  track skip 증가", 72, 254, 350, 190, 23, C.muted),
  box(72, 470, 360, 112, C.pale, true, "#C9DCF8"),
  tx("채택", 94, 491, 70, 24, 17, C.blue, { bold: true }),
  tx("log_numeric: Logistic PR-AUC +0.006798", 94, 526, 300, 35, 19, C.ink, { bold: true }),
  box(510, 166, 690, 456, C.white, true),
], "eda-preprocessing");
await addImage(s, "figures/preprocessing/05_signal_vs_noise.png", 530, 187, 650, 400, "contain", "핵심 신호와 잡음 비교");

// 5. Progression
s = makeSlide(p, [
  ...header("Baseline부터 최종 후보까지 개선 경로를 기록", "Performance Progression", 5),
  box(48, 164, 760, 470, C.white, true),
  box(840, 164, 360, 470, C.white, true),
  tx("실험 판단 원칙", 868, 192, 300, 35, 27, C.ink, { bold: true }),
  tx("채택\n• 로그 변환 수치 특징\n• 동일 fold·동일 지표 비교\n• Top 3 정밀 튜닝", 868, 245, 290, 140, 21, C.muted),
  tx("제외\n• +1 ratio 파생 특징\n  ↳ 복잡도 대비 PR-AUC 저하", 868, 415, 290, 100, 21, C.muted),
  ...pill("재학습 없이 기존 결과 재사용", 868, 550, 284, "#EAFBF5", C.green),
], "performance-progression");
await addImage(s, "figures/presentation/01_performance_progression.png", 66, 185, 724, 420, "contain", "성능 개선 진행 그래프");

// 6. Eight models
s = makeSlide(p, [
  ...header("8개 모델을 같은 조건에서 공정하게 비교", "Fair Model Comparison", 6),
  box(48, 164, 790, 470, C.white, true),
  ...metricCard("0.947622", "CatBoost baseline PR-AUC", 866, 182, 318, C.blue),
  ...metricCard("0.947534", "LightGBM baseline PR-AUC", 866, 320, 318, C.cyan),
  ...metricCard("0.947052", "XGBoost baseline PR-AUC", 866, 458, 318, C.amber),
], "eight-models");
await addImage(s, "figures/presentation/04_eight_model_fair_comparison.png", 66, 186, 754, 420, "contain", "8개 모델 동일 조건 비교");

// 7. Fine tuning
s = makeSlide(p, [
  ...header("탐색 상한을 지키고 Top 3만 정밀 튜닝", "Bounded Search", 7),
  box(48, 164, 770, 470, C.white, true),
  box(850, 164, 350, 470, C.white, true),
  tx("Fine CV PR-AUC", 878, 192, 290, 32, 24, C.muted, { bold: true }),
  tx("CatBoost", 878, 260, 170, 28, 22, C.ink, { bold: true }),
  tx("0.947913", 1040, 252, 128, 38, 28, C.blue, { bold: true, alignment: "right" }),
  tx("LightGBM", 878, 334, 170, 28, 22, C.ink, { bold: true }),
  tx("0.947806", 1040, 326, 128, 38, 28, C.cyan, { bold: true, alignment: "right" }),
  tx("XGBoost", 878, 408, 170, 28, 22, C.ink, { bold: true }),
  tx("0.947793", 1040, 400, 128, 38, 28, C.amber, { bold: true, alignment: "right" }),
  shape({ geometry: "rect", fill: C.line, position: { left: 878, top: 470 }, width: 290, height: 1 }),
  tx("차이는 작다. 따라서 단일 점수만으로 과장하지 않고 안정성·운영 해석을 함께 본다.", 878, 500, 290, 90, 19, C.muted),
], "fine-tuning");
await addImage(s, "figures/presentation/06_top3_fine_tuning_before_after.png", 66, 188, 734, 414, "contain", "Top 3 튜닝 전후 비교");

// 8. CatBoost decision
s = makeSlide(p, [
  ...header("CatBoost를 최종 위험 순위 모델로 선택", "Model Decision", 8),
  box(48, 166, 560, 446, C.white, true),
  tx("선정 근거", 76, 193, 480, 36, 28, C.ink, { bold: true }),
  tx("• Fine CV PR-AUC 1위: 0.947913\n• OOF PR-AUC: 0.947897\n• ROC-AUC: 0.941959\n• 5-seed 평균: 0.947880\n• 범주형·비선형 관계를 안정적으로 처리", 76, 252, 480, 205, 22, C.muted),
  box(76, 488, 480, 84, C.pale, true, "#C9DCF8"),
  tx("운영 기본안", 98, 508, 120, 24, 18, C.blue, { bold: true }),
  tx("Balanced F1 threshold = 0.35", 218, 504, 310, 32, 22, C.ink, { bold: true }),
  box(640, 166, 560, 446, "#FFF7E8", true, "#F2D6A3"),
  tx("중요한 해석", 668, 193, 480, 36, 28, C.amber, { bold: true }),
  tx("LightGBM은 기본 threshold에서 Recall·Calibration이 조금 더 나았다. Bootstrap 구간도 겹친다.\n\n따라서 CatBoost를 ‘절대 우승 모델’이 아니라 현재 데이터에서 근소하게 앞선 위험 순위 후보로 표현한다.", 668, 252, 480, 230, 22, C.ink),
  ...pill("모델 선택 ≠ 캠페인 효과 입증", 668, 518, 310, "#FFF0EF", C.red),
], "catboost-decision");

// 9. Threshold
s = makeSlide(p, [
  ...header("Threshold는 정답이 아니라 운영 선택지", "Operating Scenarios", 9),
  box(48, 164, 700, 470, C.white, true),
  box(780, 164, 420, 470, C.white, true),
  tx("시나리오별 trade-off", 808, 192, 360, 34, 26, C.ink, { bold: true }),
  tx("Recall 우선 · 0.29", 808, 252, 230, 26, 21, C.blue, { bold: true }),
  tx("대상 77,700 · Recall .9518 · FN 3,091", 808, 286, 350, 28, 18, C.muted),
  tx("Balanced F1 · 0.35", 808, 350, 230, 26, 21, C.green, { bold: true }),
  tx("대상 76,143 · F1 .8638 · FN 3,569", 808, 384, 350, 28, 18, C.muted),
  tx("Precision 우선 · 0.74", 808, 448, 250, 26, 21, C.amber, { bold: true }),
  tx("대상 48,463 · Precision .9507 · FP 2,388", 808, 482, 350, 28, 18, C.muted),
  tx("예산·접촉비·허용 FN에 따라 담당자가 선택", 808, 556, 350, 40, 19, C.ink, { bold: true }),
], "threshold-scenarios");
await addImage(s, "figures/presentation/10_threshold_precision_recall_f1.png", 68, 190, 660, 410, "contain", "Threshold별 Precision Recall F1");

// 10. Top-K
s = makeSlide(p, [
  ...header("Top-K로 제한된 접촉 자원의 기대 범위를 확인", "Top-K · Lift · Decile", 10),
  box(48, 164, 720, 470, C.white, true),
  ...metricCard("19.48%", "Top 10%의 이탈자 Capture", 800, 178, 384, C.blue),
  ...metricCard("100%", "Top 10% Precision (OOF)", 800, 316, 384, C.cyan),
  ...metricCard("1.948×", "Top 10% Lift", 800, 454, 384, C.green),
  tx("※ 합성 데이터의 OOF 결과이므로 실제 캠페인 성과로 해석하지 않는다.", 800, 594, 384, 34, 17, C.red),
], "topk-lift");
await addImage(s, "figures/presentation/14_topk_capture_lift.png", 66, 190, 682, 410, "contain", "Top K Capture와 Lift");

// 11. App workflow
s = makeSlide(p, [
  ...header("Streamlit에서 위험 고객을 행동 후보로 전환", "Decision Support App", 11),
  box(48, 164, 756, 470, C.white, true),
  box(836, 164, 364, 470, C.white, true),
  tx("사용 흐름", 864, 192, 310, 34, 27, C.ink, { bold: true }),
  tx("1  위험 고객 필터\n2  운영 단계·1차 신호 확인\n3  구독 유형별 비용·가치 가정\n4  접촉량·예상 규모 비교\n5  담당자 승인 후 실행", 864, 250, 292, 210, 21, C.muted),
  box(864, 490, 292, 96, "#FFF7E8", true, "#F2D6A3"),
  tx("앱이 하지 않는 것", 886, 509, 245, 24, 18, C.amber, { bold: true }),
  tx("실제 CRM 발송·할인 집행·효과 확정", 886, 544, 245, 30, 18, C.ink),
], "app-workflow");
await addImage(s, "assets/screenshots/03_customer_priority.png", 66, 184, 720, 420, "contain", "고객 우선순위 화면");

// 12. Reproducibility
s = makeSlide(p, [
  ...header("제출 패키지 자체로 재현·검토 가능하게 구성", "Reproducibility", 12),
  ...metricCard("17/17", "테스트 통과", 48, 178, 260, C.green),
  ...metricCard("6개", "Streamlit 화면 AppTest", 330, 178, 260, C.blue),
  ...metricCard("3개", "실행 완료 노트북", 612, 178, 260, C.cyan),
  ...metricCard("SHA-256", "모델 무결성 검증", 894, 178, 306, C.amber),
  box(48, 334, 552, 250, C.white, true),
  tx("패키지 구성", 74, 360, 490, 34, 26, C.ink, { bold: true }),
  tx("README · 데이터 출처 · 전처리/학습 보고서\n노트북 · 모델 · 메트릭 · 스키마\nStreamlit 앱 · 발표자료 · 요구사항 추적표", 74, 418, 480, 120, 21, C.muted),
  box(632, 334, 568, 250, C.white, true),
  tx("재현 원칙", 658, 360, 500, 34, 26, C.ink, { bold: true }),
  tx("• 기존 Run manifest와 checkpoint 재사용\n• 전체 모델 재학습 없음\n• 동일 산출물 재생성 스크립트 포함\n• 원본 CSV ↔ 그래프 추적 가능", 658, 418, 500, 125, 21, C.muted),
], "reproducibility");

// 13. Limits
s = makeSlide(p, [
  ...header("현재 결과의 경계를 밝히고 다음 검증을 제안", "Limitations & Next", 13),
  box(48, 166, 552, 446, "#FFF3F2", true, "#F2C7C3"),
  tx("지금 확정할 수 없는 것", 76, 194, 480, 34, 27, C.red, { bold: true }),
  tx("• 실제 캠페인 실행 결과\n• 이탈 방지 uplift·증분 ROI\n• 미래 시점 일반화\n• 데이터 라이선스 최종 법적 승인\n• 비용·고객가치 가정의 조직 승인", 76, 252, 480, 220, 22, C.ink),
  box(632, 166, 568, 446, "#EAFBF5", true, "#BCE8D8"),
  tx("다음 검증 우선순위", 660, 194, 500, 34, 27, C.green, { bold: true }),
  tx("1  시간 기반 외부 Holdout\n2  세그먼트별 오류·Calibration 재검증\n3  캠페인 A/B 테스트\n4  Incremental uplift·ROI 측정\n5  운영 비용·가치 가정 보정", 660, 252, 500, 220, 22, C.ink),
  ...pill("Holdout을 본 뒤 재선택하지 않기", 660, 520, 330, "#DFF7EE", C.green),
], "limitations");

// 14. Conclusion
s = makeSlide(p, [
  shape({ geometry: "rect", fill: C.navy, position: { left: 0, top: 0 }, width: W, height: H }),
  tx("FINAL TAKEAWAYS", 58, 48, 340, 24, 15, "#7DD3FC", { bold: true }),
  tx("발표에서 남길 5가지", 58, 96, 760, 62, 48, C.white, { bold: true }),
  ...[0,1,2,3,4].flatMap((i) => {
    const tops = [190, 278, 366, 454, 542];
    const msgs = [
      "EDA 신호를 전처리·모델링 판단으로 연결했다.",
      "8개 모델을 같은 조건에서 비교하고 탐색 범위를 통제했다.",
      "CatBoost는 PR-AUC·안정성 기준의 근소한 최종 후보다.",
      "Threshold·Top-K·비용 가정을 운영 선택지로 제공한다.",
      "실제 효과는 Holdout과 A/B 테스트로 검증해야 한다.",
    ];
    return [
      box(58, tops[i], 66, 58, i === 4 ? "#4B2C35" : "#203456", true, i === 4 ? "#4B2C35" : "#203456"),
      tx(String(i + 1).padStart(2, "0"), 58, tops[i] + 13, 66, 30, 21, i === 4 ? "#FCA5A5" : "#7DD3FC", { bold: true, alignment: "center" }),
      tx(msgs[i], 150, tops[i] + 8, 1000, 42, 26, C.white, { bold: true }),
    ];
  }),
  tx("PlaylistPro · Decision-ready retention analytics", 58, 670, 620, 22, 15, "#8FA5C8"),
], "final-takeaways");

await fs.mkdir(path.dirname(OUT), { recursive: true });
const file = await PresentationFile.exportPptx(p);
await file.save(OUT);
console.log(OUT);

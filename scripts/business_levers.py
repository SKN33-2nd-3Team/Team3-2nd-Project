"""Business-level analysis: lever sizing, targeting economics, segment design."""
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
pd.set_option("display.width", 220)

tr = pd.read_csv("data/train.csv"); y = tr.churned

def buckets(df):
    f = pd.DataFrame(index=df.index)
    f["hours"] = pd.cut(df.weekly_hours, [-1,5,10,40,999], labels=["h<=5","h5-10","h10-40","h>40"])
    f["skip"]  = np.where(df.song_skip_rate > .7, "skip>0.7", "skip<=0.7")
    f["pause"] = np.where(df.num_subscription_pauses >= 3, "pause>=3", "pause<=2")
    f["age"]   = pd.cut(df.age, [0,24,34,60,999], labels=["a18-24","a25-34","a35-60","a61+"])
    f["notif"] = np.where(df.notifications_clicked < 5, "notif<5", "notif>=5")
    f["plan"]  = df.subscription_type
    f["inq"]   = df.customer_service_inquiries
    return f

F = buckets(tr)
# Freeze the category set per column so an intervention that empties a level
# cannot silently shift the drop_first baseline.
CATS = {c: pd.Index(sorted(pd.Series(F[c]).astype(str).unique())) for c in F.columns}
def design(frame):
    g = pd.DataFrame({c: pd.Categorical(frame[c].astype(str), categories=CATS[c]) for c in frame.columns})
    return pd.get_dummies(g, drop_first=True).astype(float)
D = design(F)
lr = LogisticRegression(max_iter=5000, C=1e8).fit(D, y)
coef = pd.Series(lr.coef_[0], index=D.columns)
p0 = lr.predict_proba(D)[:, 1]
print(f"baseline mean churn prob = {p0.mean():.4f}\n")

# ---------- 1. LEVER SIZING: counterfactual on the addressable population ----------
print("="*90)
print("1. LEVER SIZING  (counterfactual: apply lever, recompute churn prob)")
print("="*90)
levers = {
    "A. Free -> Family (유료 전환)":        ("plan",  tr.subscription_type == "Free", {"plan": "Family"}),
    "B. Student -> Family":                  ("plan",  tr.subscription_type == "Student", {"plan": "Family"}),
    "C. 청취 <=5h -> 5-10h (1단계 회복)":   ("hours", tr.weekly_hours <= 5, {"hours": "h5-10"}),
    "D. 청취 <=10h -> 10-40h (정상권 진입)": ("hours", tr.weekly_hours <= 10, {"hours": "h10-40"}),
    "E. skip>0.7 -> <=0.7 (추천 개선)":     ("skip",  tr.song_skip_rate > .7, {"skip": "skip<=0.7"}),
    "F. 상담 High -> Medium (응대 개선)":   ("inq",   tr.customer_service_inquiries == "High", {"inq": "Medium"}),
    "G. 상담 High -> Low":                   ("inq",   tr.customer_service_inquiries == "High", {"inq": "Low"}),
    "H. 알림클릭 <5 -> >=5 (커뮤니케이션)": ("notif", tr.notifications_clicked < 5, {"notif": "notif>=5"}),
    "I. 일시정지 >=3 -> <=2 (정지 방지)":   ("pause", tr.num_subscription_pauses >= 3, {"pause": "pause<=2"}),
}
rows = []
for name, (_, mask, change) in levers.items():
    Fc = F.copy()
    for col, val in change.items():
        Fc.loc[mask, col] = val
    Dc = design(Fc).reindex(columns=D.columns, fill_value=0)
    p1 = lr.predict_proba(Dc)[:, 1]
    n = int(mask.sum())
    saved = (p0[mask.to_numpy()] - p1[mask.to_numpy()])
    rows.append({
        "lever": name, "대상자수": n, "대상비중": f"{mask.mean():.1%}",
        "대상_이탈확률_전": p0[mask.to_numpy()].mean(),
        "대상_이탈확률_후": p1[mask.to_numpy()].mean(),
        "1인당_감소pp": saved.mean()*100,
        "총_이탈방지_명": saved.sum(),
    })
r = pd.DataFrame(rows).sort_values("총_이탈방지_명", ascending=False)
print(r.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

# ---------- 2. WHERE does a lever pay off most? (dp/dlogit is max at p=0.5) ----------
print("\n" + "="*90)
print("2. 같은 레버, 다른 위치 - 위험도별 실제 효과 (Free->Family, +/-4.5 logit)")
print("="*90)
free = tr.subscription_type == "Free"
Fc = F.copy(); Fc.loc[free, "plan"] = "Family"
Dc = design(Fc).reindex(columns=D.columns, fill_value=0)
p1 = lr.predict_proba(Dc)[:, 1]
sub = pd.DataFrame({"p_before": p0[free.to_numpy()], "p_after": p1[free.to_numpy()]})
sub["감소pp"] = (sub.p_before - sub.p_after) * 100
sub["위험대"] = pd.cut(sub.p_before, [0,.5,.7,.85,.95,1.0],
                      labels=["<50%","50-70%","70-85%","85-95%","95%+"])
g = sub.groupby("위험대", observed=True).agg(대상자수=("p_before","size"), 평균_전=("p_before","mean"),
                                             평균_후=("p_after","mean"), 평균감소pp=("감소pp","mean"))
g["총방지명"] = g.대상자수 * g.평균감소pp / 100
print(g.to_string(float_format=lambda v: f"{v:,.1f}"))

# ---------- 3. 3-tier segment design ----------
print("\n" + "="*90)
print("3. 깃발 기반 3단계 세그먼트 (운영 설계)")
print("="*90)
flags = ((tr.weekly_hours <= 10).astype(int) + (tr.song_skip_rate > .7).astype(int)
         + (tr.num_subscription_pauses >= 3).astype(int) + (tr.customer_service_inquiries == "High").astype(int)
         + (tr.subscription_type == "Free").astype(int))
tier = pd.cut(flags, [-1,1,2,5], labels=["T3 방치(0-1개)","T2 자동화(2개)","T1 집중(3개+)"])
t = pd.concat([tier.rename("tier"), y], axis=1).groupby("tier", observed=True)["churned"].agg(["size","mean","sum"])
t.columns = ["고객수","이탈률","이탈자수"]
t["전체대비_고객"] = t.고객수/len(tr); t["이탈자_점유"] = t.이탈자수/y.sum()
print(t.to_string(float_format=lambda v: f"{v:,.3f}"))

# ---------- 4. campaign economics ----------
print("\n" + "="*90)
print("4. 캠페인 손익분기 (가정: 월 ARPU 10,000원, 잔존 12개월 => LTV 120,000원)")
print("="*90)
LTV, base = 120_000, y.mean()
sat = pd.concat([F, y], axis=1).groupby(list(F.columns), observed=True)["churned"].transform("mean")
order = np.argsort(-sat.to_numpy())
for k in (10, 20, 30, 50):
    idx = order[:int(len(tr)*k/100)]
    churners = y.to_numpy()[idx].sum()
    prec = churners/len(idx)
    for eff in (0.10, 0.20):
        gain = churners * eff * LTV
        print(f"  상위{k:3d}% ({len(idx):6,}명, 정밀도{prec:.1%}) | 캠페인성공률{eff:.0%} "
              f"| 방지 {churners*eff:6,.0f}명 | 기대이익 {gain/1e8:5.2f}억 "
              f"| 손익분기 1인당비용 {gain/len(idx):6,.0f}원")

# ---------- 5. actionability ----------
print("\n" + "="*90)
print("5. 신호 7개의 실행가능성 분류")
print("="*90)
act = {
 "weekly_hours":"실행가능 - 추천/복귀알림/큐레이션", "subscription_type":"실행가능 - 요금제 전환 오퍼",
 "customer_service_inquiries":"실행가능 - 응대품질/셀프서비스", "song_skip_rate":"실행가능 - 추천 알고리즘 개선",
 "num_subscription_pauses":"실행가능 - 정지 시점 개입", "notifications_clicked":"실행가능 - 채널/빈도 최적화",
 "age":"실행불가 - 타겟팅에만 사용"}
for k, v in act.items(): print(f"  {k:28s} {v}")

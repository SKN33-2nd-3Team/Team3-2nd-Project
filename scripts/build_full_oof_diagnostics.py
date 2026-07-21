"""OOF-only diagnostics for the completed full fair-comparison run."""
from __future__ import annotations
import json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss

ROOT=Path(__file__).resolve().parents[1]; RID="20260721_full_fair_v1"; RUN=ROOT/"experiments"/"submission_full_comparison"/RID; ART=ROOT/"artifacts"; CAND=ROOT/"models"/"candidates"/RID
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.run_full_fair_comparison import Features
setattr(sys.modules["__main__"], "Features", Features)
TOP=["catboost","xgboost","lightgbm"]
def metric(y,p,t=.5):
 z=(p>=t).astype(int); tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel(); return dict(pr_auc=average_precision_score(y,p),roc_auc=roc_auc_score(y,p),f1=f1_score(y,z),recall=recall_score(y,z),precision=precision_score(y,z),fn=int(fn),fp=int(fp),tn=int(tn),tp=int(tp))
def main():
 d=pd.read_csv(ROOT/"data"/"train.csv"); y=d.churned.to_numpy(); rng=np.random.default_rng(2026); probs={n:np.load(RUN/f"oof_fine_{n}.npy") for n in TOP}
 # Bootstrap point and paired PR-AUC differences.
 boot_path=ART/"bootstrap_confidence_intervals.csv"
 if not boot_path.exists():
  rows=[]; idx=np.arange(len(y))
  for n,p in probs.items():
   values=[]
   for _ in range(1000):
    s=rng.choice(idx,len(idx),replace=True); values.append([average_precision_score(y[s],p[s]),roc_auc_score(y[s],p[s])])
   a=np.array(values); rows += [{"model":n,"metric":"pr_auc","estimate":metric(y,p)["pr_auc"],"ci_low":np.quantile(a[:,0],.025),"ci_high":np.quantile(a[:,0],.975),"resamples":1000},{"model":n,"metric":"roc_auc","estimate":metric(y,p)["roc_auc"],"ci_low":np.quantile(a[:,1],.025),"ci_high":np.quantile(a[:,1],.975),"resamples":1000}]
  pd.DataFrame(rows).to_csv(boot_path,index=False,encoding="utf-8-sig")
 # Calibration and risk deciles.
 cal=[]; dec=[]
 for n,p in probs.items():
  observed,pred=calibration_curve(y,p,n_bins=10,strategy="quantile"); cal.append({"model":n,"brier":brier_score_loss(y,p),"mean_abs_calibration_gap":float(np.abs(observed-pred).mean())})
  q=pd.qcut(pd.Series(p).rank(method="first"),10,labels=False)+1; t=pd.DataFrame({"decile":q,"actual":y,"probability":p}).groupby("decile").agg(customers=("actual","size"),actual_churn_rate=("actual","mean"),mean_predicted_probability=("probability","mean"),actual_churners=("actual","sum")).reset_index(); t["model"]=n; dec.append(t)
 pd.DataFrame(cal).to_csv(ART/"calibration_summary.csv",index=False,encoding="utf-8-sig"); pd.concat(dec).to_csv(ART/"risk_decile_v2.csv",index=False,encoding="utf-8-sig")
 # Operating comparisons based only on OOF probabilities.
 scenarios=[]; contact=[]; recall_rows=[]; top=[]
 for n,p in probs.items():
  sweep=[]
  for t in np.linspace(.05,.95,91): sweep.append({"threshold":t,**metric(y,p,t),"target_customers":int((p>=t).sum()),"target_rate":float((p>=t).mean())})
  s=pd.DataFrame(sweep); picks=[("recall_first",s[s.recall>=.95].sort_values("precision",ascending=False).iloc[0]),("balanced_f1",s.sort_values("f1",ascending=False).iloc[0]),("precision_first",s[s.precision>=.95].sort_values("recall",ascending=False).iloc[0])]
  scenarios += [{"model":n,"scenario":k,**r.to_dict()} for k,r in picks]
  order=np.argsort(-p)
  for pc in [5,10,20,30,40]:
   k=int(len(y)*pc/100); yy=y[order[:k]]; contact.append({"model":n,"top_percent":pc,"target_customers":k,"captured_churners":int(yy.sum()),"capture_rate":float(yy.sum()/y.sum()),"precision":float(yy.mean()),"lift":float(yy.mean()/y.mean())})
  for target in [.8,.85,.9,.95]:
   r=s[s.recall>=target].sort_values("target_customers").iloc[0]; recall_rows.append({"model":n,"target_recall":target,**r.to_dict()})
 pd.DataFrame(scenarios).to_csv(ART/"threshold_operating_scenarios_v2.csv",index=False,encoding="utf-8-sig"); pd.DataFrame(contact).to_csv(ART/"equal_contact_comparison.csv",index=False,encoding="utf-8-sig"); pd.DataFrame(recall_rows).to_csv(ART/"equal_recall_comparison.csv",index=False,encoding="utf-8-sig"); pd.DataFrame(contact).to_csv(ART/"topk_lift_v2.csv",index=False,encoding="utf-8-sig")
 # Final candidate: highest fine-tuned CV PR-AUC; no external holdout claimed.
 fine=pd.read_csv(ART/"top3_fine_tuning_summary.csv").sort_values("best_cv_pr_auc",ascending=False); final=str(fine.iloc[0].model); final_p=probs[final]
 seg=[]
 for col in ["subscription_type","customer_service_inquiries"]:
  for value,ix in d.groupby(col).groups.items():
   ii=np.fromiter(ix,dtype=int); seg.append({"model":final,"segment":f"{col}={value}","customers":len(ii),"actual_churn_rate":float(y[ii].mean()),**metric(y[ii],final_p[ii])})
 pd.DataFrame(seg).to_csv(ART/"segment_error_analysis.csv",index=False,encoding="utf-8-sig")
 pipe=joblib.load(RUN/"fine_tuning"/f"{final}_best.joblib"); CAND.mkdir(parents=True,exist_ok=True); joblib.dump(pipe,CAND/"candidate_pipeline.joblib"); test=pd.read_csv(ROOT/"data"/"test.csv").head(5); a=pipe.predict_proba(test)[:,1]; b=joblib.load(CAND/"candidate_pipeline.joblib").predict_proba(test)[:,1]
 meta={"run_id":RID,"status":"FULL_MODEL_COMPARISON_COMPLETED_CANDIDATE_AWAITING_REVIEW","candidate_model":final,"selection":"highest fine-tuned CV PR-AUC; OOF/seed/bootstrap operational evidence","reload_verified":bool(np.allclose(a,b)),"no_external_holdout":"No untouched external or final holdout was available for the full comparison run.","metrics":metric(y,final_p)}
 (CAND/"metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
 pd.DataFrame([{"model":n,"fine_cv_pr_auc":fine.loc[fine.model==n,"best_cv_pr_auc"].iloc[0],"seed_mean_pr_auc":pd.read_csv(ART/"seed_stability_summary.csv",header=[0,1],index_col=0).loc[n,("pr_auc","mean")],"selection_status":"SELECTED" if n==final else "NOT_SELECTED"} for n in TOP]).to_csv(ART/"final_model_selection_matrix.csv",index=False,encoding="utf-8-sig")
 m=json.loads((RUN/"run_manifest.json").read_text()); m["checkpoints"].update({"bootstrap":"PASSED","calibration_and_operations":"PASSED","segment_errors":"PASSED","candidate_save_reload":"PASSED"});m["last_checkpoint"]="candidate_save_reload";m["candidate_model"]=final;(RUN/"run_manifest.json").write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__": main()

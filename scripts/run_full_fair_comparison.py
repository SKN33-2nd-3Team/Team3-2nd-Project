"""Checkpointed full fair comparison. It never reads P0 predictions for selection."""
from __future__ import annotations

import argparse, json, os, shutil, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import joblib, numpy as np, pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict, RandomizedSearchCV, ParameterSampler, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT=Path(__file__).resolve().parents[1]; RID="20260721_full_fair_v1"
RUN=ROOT/"experiments"/"submission_full_comparison"/RID; CAND=ROOT/"models"/"candidates"/RID
SEED=42; FOLDS=5; VARIANTS=["raw","plus1_ratios","zero_aware_ratios","log_numeric","interaction","signal_pruned"]

class Features(BaseEstimator,TransformerMixin):
    def __init__(self,variant="plus1_ratios"): self.variant=variant
    def fit(self,X,y=None): return self
    def transform(self,X):
        d=X.copy().drop(columns=["customer_id"],errors="ignore")
        signup=pd.to_numeric(d["signup_date"],errors="coerce"); d["signup_days_ago"]=-signup; d=d.drop(columns=["signup_date"])
        if self.variant!="raw":
            pairs={"unique_song_ratio":("weekly_unique_songs","weekly_songs_played"),"shared_playlist_ratio":("num_shared_playlists","num_playlists_created"),"hours_per_song":("weekly_hours","weekly_songs_played"),"friends_per_playlist":("num_platform_friends","num_playlists_created")}
            for name,(num,den) in pairs.items():
                if self.variant=="zero_aware_ratios":
                    d[name]=np.where(d[den].eq(0),0.,d[num]/d[den]); d[f"{den}_is_zero"]=d[den].eq(0).astype(int)
                else: d[name]=d[num]/(d[den]+1.)
        if self.variant=="log_numeric":
            for c in ["weekly_hours","weekly_songs_played","weekly_unique_songs","num_platform_friends","num_playlists_created","num_shared_playlists"]: d[f"log1p_{c}"]=np.log1p(d[c].clip(lower=0))
        if self.variant=="interaction": d["plan_x_inquiry"]=d["subscription_type"].astype(str)+"__"+d["customer_service_inquiries"].astype(str)
        if self.variant=="signal_pruned":
            keep=["weekly_hours","subscription_type","customer_service_inquiries","num_subscription_pauses","song_skip_rate","age","notifications_clicked","signup_days_ago"]
            d=d[[c for c in keep if c in d]]
        return d.replace([np.inf,-np.inf],np.nan)

def manifest(): return json.loads((RUN/"run_manifest.json").read_text(encoding="utf-8"))
def save_manifest(m): (RUN/"run_manifest.json").write_text(json.dumps(m,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

def _top3() -> list[str]:
    path=ROOT/"artifacts"/"top3_selection_matrix.csv"
    if path.exists():
        return pd.read_csv(path).sort_values("rank").model.astype(str).tolist()
    try:
        return [str(name) for name in manifest().get("top3",[])]
    except (FileNotFoundError,json.JSONDecodeError):
        return []

def required_outputs(step: str) -> list[Path]:
    static={
      "fair_fold_assignment":[RUN/"cv_fold_assignments.csv",*[RUN/f"fold_{i}.npz" for i in range(FOLDS)]],
      "insight_inventory":[ROOT/"artifacts"/"current_insight_inventory.csv",ROOT/"reports"/"current_insight_inventory.md"],
      "preprocessing_experiments":[ROOT/"artifacts"/"preprocessing_experiment_results.csv",ROOT/"artifacts"/"insight_preprocessing_register.csv",ROOT/"reports"/"insight_driven_preprocessing_review.md",*[RUN/f"oof_feature_{v}_logistic.npy" for v in VARIANTS]],
      "eight_model_baseline":[ROOT/"artifacts"/"model_comparison_fair.csv",*[RUN/"baseline"/f"{name}.json" for name in models()],*[RUN/f"oof_{name}.npy" for name in models()]],
      "final_common_feature_set":[RUN/"tree_feature_validation.csv",RUN/"feature_selection_summary.csv"],
      "seven_model_random_search":[ROOT/"artifacts"/"random_search_summary.csv",*[RUN/"random_search"/f"{name}_random_search.csv" for name in models() if name!="dummy"],*[RUN/"random_search"/f"{name}_best.joblib" for name in models() if name!="dummy"],*[RUN/f"oof_search_{name}.npy" for name in models() if name!="dummy"]],
      "top3_selection":[ROOT/"artifacts"/"top3_selection_matrix.csv",ROOT/"reports"/"top3_selection_decision.md"],
      "seed_stability":[ROOT/"artifacts"/"seed_stability_summary.csv",RUN/"seed_stability_all.csv"],
    }
    if step=="top3_fine_tuning":
        top=_top3()
        if len(top)!=3:
            return [ROOT/"artifacts"/"top3_selection_matrix.csv"]
        return [ROOT/"artifacts"/"top3_fine_tuning_summary.csv",*[RUN/"fine_tuning"/f"{name}_fine_tuning.csv" for name in top],*[RUN/"fine_tuning"/f"{name}_best.joblib" for name in top],*[RUN/f"oof_fine_{name}.npy" for name in top]]
    if step=="seed_stability":
        top=_top3()
        return static[step]+[RUN/"seed_stability"/f"{name}_seeds.csv" for name in top]
    return static.get(step,[])

def done(step):
    m=manifest()
    return m["checkpoints"].get(step)=="PASSED" and all(path.is_file() for path in required_outputs(step))
def mark(step,status="PASSED",**extra):
    m=manifest(); m["checkpoints"][step]=status; m["last_checkpoint"]=step; m.update(extra); save_manifest(m)
def load():
    d=pd.read_csv(ROOT/"data"/"train.csv"); return d.drop(columns="churned"),d.churned.astype(int)
def folds(y):
    out=RUN/"cv_fold_assignments.csv"
    fold_files=[RUN/f"fold_{i}.npz" for i in range(FOLDS)]
    if out.exists() and all(path.exists() for path in fold_files):
        loaded=[]
        for path in fold_files:
            with np.load(path) as data:
                loaded.append((data["train"],data["valid"]))
        if len(pd.read_csv(out))==len(y):
            return loaded
    cv=StratifiedKFold(FOLDS,shuffle=True,random_state=SEED); arr=[]
    for i,(tr,va) in enumerate(cv.split(np.zeros(len(y)),y)): np.savez(RUN/f"fold_{i}.npz",train=tr,valid=va); arr.append((tr,va))
    f=np.empty(len(y),int)
    for i,(_,va) in enumerate(arr): f[va]=i
    pd.DataFrame({"row_id":np.arange(len(y)),"fold":f}).to_csv(out,index=False); return arr

def prepare_fresh_run() -> Path:
    """Archive generated run state and reset checkpoints without deleting evidence."""
    RUN.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive=ROOT/"tmp"/"full_fair_runs"/f"{RID}_{stamp}"
    archive.mkdir(parents=True,exist_ok=False)
    keep={"execution_plan.md","run_config.json"}
    for path in list(RUN.iterdir()):
        if path.name in keep:
            continue
        shutil.move(str(path),archive/path.name)
    fresh={
      "run_id":RID,
      "status":"IN_PROGRESS",
      "last_checkpoint":None,
      "checkpoints":{"repository_and_data_audit":"PASSED"},
      "completed_models":[],
      "completed_trials":{},
      "blocked_reason":None,
      "resume_command":"python scripts/run_full_fair_comparison.py --stage all",
      "external_labeled_holdout":"NOT_AVAILABLE",
      "retraining_performed_during_completion":True,
    }
    save_manifest(fresh)
    return archive
def build(X,model,variant,scale=False):
    sample=Features(variant).transform(X.iloc[:10]); nums=sample.select_dtypes(include=np.number).columns.tolist(); cats=[c for c in sample.columns if c not in nums]
    ns=[("impute",SimpleImputer(strategy="median"))]+([("scale",StandardScaler())] if scale else [])
    pre=ColumnTransformer([("num",Pipeline(ns),nums),("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cats)],verbose_feature_names_out=False)
    return Pipeline([("features",Features(variant)),("pre",pre),("model",model)])
def models():
    return {
      "dummy":(DummyClassifier(strategy="prior"),False),"logistic":(LogisticRegression(max_iter=1800,random_state=SEED),True),
      "decision_tree":(DecisionTreeClassifier(max_depth=10,min_samples_leaf=20,random_state=SEED),False),
      "random_forest":(RandomForestClassifier(n_estimators=100,max_depth=14,min_samples_leaf=5,n_jobs=4,random_state=SEED),False),
      "gradient_boosting":(GradientBoostingClassifier(n_estimators=150,learning_rate=.05,max_depth=3,random_state=SEED),False),
      "xgboost":(XGBClassifier(n_estimators=150,learning_rate=.05,max_depth=3,subsample=.9,colsample_bytree=.9,n_jobs=4,eval_metric="logloss",random_state=SEED),False),
      "lightgbm":(LGBMClassifier(n_estimators=150,learning_rate=.05,num_leaves=31,subsample=.9,colsample_bytree=.9,n_jobs=4,verbosity=-1,random_state=SEED),False),
      "catboost":(CatBoostClassifier(iterations=150,learning_rate=.05,depth=6,verbose=False,thread_count=4,random_seed=SEED),False)}
def score(y,p):
    z=(p>=.5).astype(int); tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel()
    return {"pr_auc":average_precision_score(y,p),"roc_auc":roc_auc_score(y,p),"brier":brier_score_loss(y,p),"f1":f1_score(y,z),"recall":recall_score(y,z),"precision":precision_score(y,z),"fn":int(fn),"fp":int(fp),"tp":int(tp),"tn":int(tn)}
def oof(X,y,pipe,cv): return cross_val_predict(pipe,X,y,cv=cv,method="predict_proba",n_jobs=1)[:,1]
def write_oof(name,p): np.save(RUN/f"oof_{name}.npy",p)

def setup():
    X,y=load(); folds(y); mark("fair_fold_assignment");
    rows=[
      ["I01","Low weekly listening and high skip rate are associated with label","EDA rate charts / feature importance","VALIDATED_ASSOCIATION","weekly_hours,song_skip_rate","Prioritize diagnostic segment; no causal claim","No time ordering"],
      ["I02","Free subscription has high observed churn","EDA categorical rates","VALIDATED_ASSOCIATION","subscription_type","Offer hypothesis only","Offer effect unknown"],
      ["I03","High service inquiries are associated with label","EDA categorical rates","VALIDATED_ASSOCIATION","customer_service_inquiries","Service-resolution experiment hypothesis","Association only"],
      ["I04","Pauses and age add predictive signal","feature_importance.csv","MODEL_SUPPORTED","num_subscription_pauses,age","Use age only with fairness review","Age actionability restricted"],
      ["I05","Top-K risk ranking captures observed positives","P0 targeting artifacts","MODEL_SUPPORTED","model probability","Capacity planning only","P0 reference, not selection evidence"],
    ]
    rows=[r[:3]+["repository audit / named artifact"]+r[3:] for r in rows]
    inv=pd.DataFrame(rows,columns=["insight_id","insight","evidence","evidence_file","status","related_features","actionability","limitation"]); inv.to_csv(ROOT/"artifacts"/"current_insight_inventory.csv",index=False,encoding="utf-8-sig")
    (ROOT/"reports"/"current_insight_inventory.md").write_text("# Current Insight Inventory\n\n```text\n"+inv.to_string(index=False)+"\n```\n\nAll associations are non-causal.\n",encoding="utf-8"); mark("insight_inventory")

def feature_experiments():
    if done("preprocessing_experiments"): return
    X,y=load(); cv=folds(y); rows=[]
    for v in VARIANTS:
        for name in ["logistic"]:
            model,sc=models()[name]; t=time.time(); p=oof(X,y,build(X,model,v,sc),cv); met=score(y,p); write_oof(f"feature_{v}_{name}",p)
            rows.append({"experiment_id":f"FE_{v}","insight_id":"I01/I02/I03","variant":v,"validator":name,"seconds":time.time()-t,**met,"decision":"pending common feature selection"})
    tab=pd.DataFrame(rows); base=tab.loc[tab.variant=="raw","pr_auc"].iloc[0]; tab["delta_pr_auc_vs_raw"]=tab.pr_auc-base
    tab.to_csv(ROOT/"artifacts"/"preprocessing_experiment_results.csv",index=False,encoding="utf-8-sig")
    best=tab.sort_values("pr_auc",ascending=False).iloc[0].variant
    register=pd.DataFrame([{"variant":v,"feature_change":v,"prediction_time_available":True,"leakage_risk":"none; deterministic raw inputs only","logistic_pr_auc":tab.loc[tab.variant==v,"pr_auc"].iloc[0],"decision":"provisional_adopt" if v==best else "exclude","reason":"highest logistic OOF PR-AUC" if v==best else "lower logistic OOF PR-AUC"} for v in VARIANTS])
    register.to_csv(ROOT/"artifacts"/"insight_preprocessing_register.csv",index=False,encoding="utf-8-sig")
    (ROOT/"reports"/"insight_driven_preprocessing_review.md").write_text("# Insight-Driven Preprocessing Review\n\n```text\n"+register.to_string(index=False)+"\n```\n\nRatio variants compare +1 smoothing, zero-aware indicators, and transformed alternatives; no target-derived encoding is used.\n",encoding="utf-8")
    mark("preprocessing_experiments",provisional_feature_variant=str(best))

def baseline():
    if done("eight_model_baseline"): return
    X,y=load(); cv=folds(y); v=manifest()["provisional_feature_variant"]; rows=[]; out=RUN/"baseline"; out.mkdir(exist_ok=True); only=os.environ.get("FULL_MODEL")
    for name,(model,sc) in models().items():
        if only and name!=only: continue
        saved=out/f"{name}.json"
        oof_file=RUN/f"oof_{name}.npy"
        if saved.exists() and oof_file.exists(): row=json.loads(saved.read_text(encoding="utf-8"))
        elif oof_file.exists():
            p=np.load(oof_file); row={"model":name,"feature_variant":v,"cv_folds":5,"seconds":np.nan,**score(y,p)}; saved.write_text(json.dumps(row,indent=2),encoding="utf-8")
        else:
            t=time.time(); p=oof(X,y,build(X,model,v,sc),cv); write_oof(name,p); row={"model":name,"feature_variant":v,"cv_folds":5,"seconds":time.time()-t,**score(y,p)}; saved.write_text(json.dumps(row,indent=2),encoding="utf-8")
        rows.append(row)
        m=manifest(); m["completed_models"]=sorted(set(m.get("completed_models",[])+[name])); save_manifest(m)
    allrows=[json.loads(p.read_text(encoding="utf-8")) for p in out.glob("*.json")]
    if len(allrows)==8:
        tab=pd.DataFrame(allrows).sort_values("pr_auc",ascending=False); tab.to_csv(ROOT/"artifacts"/"model_comparison_fair.csv",index=False,encoding="utf-8-sig")
        tree=tab[tab.model.isin(["gradient_boosting","lightgbm","catboost"])].iloc[0].model; mark("eight_model_baseline",tree_validator=str(tree))

def tree_feature_validation():
    if done("final_common_feature_set"): return
    X,y=load(); cv=folds(y); name=manifest()["tree_validator"]; model,sc=models()[name]; rows=[]
    for v in VARIANTS:
        t=time.time(); p=oof(X,y,build(X,model,v,sc),cv); rows.append({"experiment_id":f"FE_TREE_{v}","variant":v,"validator":name,"seconds":time.time()-t,**score(y,p)})
    path=RUN/"tree_feature_validation.csv"; pd.DataFrame(rows).to_csv(path,index=False)
    log=pd.read_csv(ROOT/"artifacts"/"preprocessing_experiment_results.csv"); tree=pd.DataFrame(rows); combined=log.groupby("variant").pr_auc.mean().rename("logistic_pr_auc").to_frame().join(tree.set_index("variant").pr_auc.rename("tree_pr_auc")); combined["mean_pr_auc"]=combined.mean(axis=1); final=combined.sort_values("mean_pr_auc",ascending=False).index[0]
    combined.to_csv(RUN/"feature_selection_summary.csv"); mark("final_common_feature_set",final_feature_variant=str(final))

def search_space(name):
    s={"logistic":{"model__C":np.logspace(-3,2,30),"model__class_weight":[None,"balanced"],"model__penalty":["l2"]},"decision_tree":{"model__max_depth":[3,5,7,10,14,None],"model__min_samples_leaf":[1,5,10,20,40],"model__criterion":["gini","entropy"]},"random_forest":{"model__n_estimators":[10,20,30],"model__max_depth":[4,6,8],"model__min_samples_leaf":[20,40,80],"model__max_features":["sqrt",.4]},"gradient_boosting":{"model__n_estimators":[40,60,80],"model__learning_rate":[.04,.07,.1],"model__max_depth":[1,2,3],"model__min_samples_leaf":[20,40,80]},"xgboost":{"model__n_estimators":[100,150,200],"model__learning_rate":[.02,.04,.06,.1],"model__max_depth":[2,3,4,5],"model__subsample":[.7,.85,1.],"model__colsample_bytree":[.7,.85,1.]},"lightgbm":{"model__n_estimators":[100,150,200],"model__learning_rate":[.02,.04,.06,.1],"model__num_leaves":[15,31,63],"model__min_child_samples":[10,20,40],"model__subsample":[.7,.85,1.]},"catboost":{"model__iterations":[100,150,200],"model__learning_rate":[.02,.04,.06,.1],"model__depth":[4,5,6,8],"model__l2_leaf_reg":[1,3,5,8]}}
    return s[name]

def randomized():
    if done("seven_model_random_search"): return
    X,y=load(); cv=folds(y); v=manifest()["final_feature_variant"]; cfg=json.loads((RUN/"run_config.json").read_text()); out=RUN/"random_search"; out.mkdir(exist_ok=True); rows=[]
    only=os.environ.get("FULL_MODEL")
    for name,(model,sc) in models().items():
        if name=="dummy": continue
        if only and name!=only: continue
        result=out/f"{name}_random_search.csv"; bestfile=out/f"{name}_best.joblib"
        if result.exists() and bestfile.exists():
            rs=pd.read_csv(result); best=joblib.load(bestfile)
        elif name in {"random_forest", "gradient_boosting"}:
            # Checkpoint each sampled trial: identical 24-trial/5-fold protocol, resilient to command time limits.
            cv_jobs = 1 if name == "random_forest" else max(
                1, min(5, int(os.environ.get("FULL_CV_JOBS", "1")))
            )
            params_file=out/f"{name}_sampled_params.json"; partial=out/f"{name}_partial_trials.csv"
            samples=json.loads(params_file.read_text()) if params_file.exists() else list(ParameterSampler(search_space(name),n_iter=cfg["random_search_min_trials"][name],random_state=SEED))
            if not params_file.exists(): params_file.write_text(json.dumps(samples,default=lambda x: int(x) if isinstance(x,np.integer) else float(x)),encoding="utf-8")
            prior=pd.read_csv(partial) if partial.exists() else pd.DataFrame(); limit=int(os.environ.get("FULL_TRIALS_PER_INVOCATION","24")); start=len(prior); add=[]
            for trial in range(start,min(start+limit,len(samples))):
                candidate=clone(build(X,model,v,sc)).set_params(**samples[trial]); started=time.time()
                try:
                 values=cross_val_score(candidate,X,y,scoring="average_precision",cv=cv,n_jobs=cv_jobs); add.append({"trial":trial,"params_json":json.dumps(samples[trial]),"mean_test_score":float(values.mean()),"std_test_score":float(values.std()),"status":"success","seconds":time.time()-started})
                except Exception as exc: add.append({"trial":trial,"params_json":json.dumps(samples[trial]),"mean_test_score":np.nan,"std_test_score":np.nan,"status":f"failed:{type(exc).__name__}","seconds":time.time()-started})
                pd.concat([prior,pd.DataFrame(add)],ignore_index=True).to_csv(partial,index=False)
            rs=pd.read_csv(partial)
            if len(rs)<len(samples): return
            rs.to_csv(result,index=False); winner=json.loads(rs.loc[rs.mean_test_score.idxmax(),"params_json"]); best=clone(build(X,model,v,sc)).set_params(**winner).fit(X,y); joblib.dump(best,bestfile)
        else:
            q=RandomizedSearchCV(build(X,model,v,sc),search_space(name),n_iter=cfg["random_search_min_trials"][name],scoring="average_precision",cv=cv,n_jobs=1,random_state=SEED,refit=True,error_score=np.nan,return_train_score=True)
            q.fit(X,y); rs=pd.DataFrame(q.cv_results_); rs.to_csv(result,index=False); best=q.best_estimator_; joblib.dump(best,bestfile)
        p=oof(X,y,best,cv); write_oof(f"search_{name}",p); rows.append({"model":name,"trials":len(rs),"valid_trials":int(rs.mean_test_score.notna().sum()),"best_cv_pr_auc":float(np.nanmax(rs.mean_test_score)),"cv_pr_auc_std":float(rs.loc[rs.mean_test_score.idxmax(),"std_test_score"]),**score(y,p)})
        m=manifest(); m["completed_trials"][name]=int(len(rs)); save_manifest(m)
    allrows=[]
    for p in out.glob("*_random_search.csv"):
        name=p.name.replace("_random_search.csv",""); best=joblib.load(out/f"{name}_best.joblib") if (out/f"{name}_best.joblib").exists() else None
        if best is not None:
            rs=pd.read_csv(p); prob=np.load(RUN/f"oof_search_{name}.npy") if (RUN/f"oof_search_{name}.npy").exists() else None
            if prob is not None: allrows.append({"model":name,"trials":len(rs),"valid_trials":int(rs.mean_test_score.notna().sum()),"best_cv_pr_auc":float(np.nanmax(rs.mean_test_score)),"cv_pr_auc_std":float(rs.loc[rs.mean_test_score.idxmax(),"std_test_score"]),**score(y,prob)})
    if len(allrows)==7:
        pd.DataFrame(allrows).sort_values("best_cv_pr_auc",ascending=False).to_csv(ROOT/"artifacts"/"random_search_summary.csv",index=False,encoding="utf-8-sig"); mark("seven_model_random_search")

def fine_tune():
    if done("top3_fine_tuning"): return
    X,y=load(); cv=folds(y); v=manifest()["final_feature_variant"]; summ=pd.read_csv(ROOT/"artifacts"/"random_search_summary.csv").sort_values(["best_cv_pr_auc","cv_pr_auc_std"],ascending=[False,True]); top=summ.head(3).model.tolist(); pd.DataFrame({"rank":range(1,4),"model":top,"reason":"actual randomized-search CV ranking"}).to_csv(ROOT/"artifacts"/"top3_selection_matrix.csv",index=False)
    (ROOT/"reports"/"top3_selection_decision.md").write_text("# Top 3 Selection\n\n```text\n"+summ.head(3).to_string(index=False)+"\n```\n",encoding="utf-8"); out=RUN/"fine_tuning"; out.mkdir(exist_ok=True); rows=[]
    only=os.environ.get("FULL_MODEL")
    for name in top:
        if only and name!=only: continue
        result=out/f"{name}_fine_tuning.csv"; best_file=out/f"{name}_best.joblib"
        if result.exists() and best_file.exists(): rs=pd.read_csv(result); best=joblib.load(best_file)
        else:
            base=joblib.load(RUN/"random_search"/f"{name}_best.joblib"); q=RandomizedSearchCV(base,search_space(name),n_iter=15,scoring="average_precision",cv=cv,n_jobs=1,random_state=2026,refit=True,error_score=np.nan,return_train_score=True); q.fit(X,y); rs=pd.DataFrame(q.cv_results_); rs.to_csv(result,index=False); best=q.best_estimator_; joblib.dump(best,best_file)
        oof_file=RUN/f"oof_fine_{name}.npy"; p=np.load(oof_file) if oof_file.exists() else oof(X,y,best,cv)
        if not oof_file.exists(): write_oof(f"fine_{name}",p)
        rows.append({"model":name,"trials":len(rs),"valid_trials":int(rs.mean_test_score.notna().sum()),"best_cv_pr_auc":float(np.nanmax(rs.mean_test_score)),**score(y,p)})
    allrows=[]
    for name in top:
        result=out/f"{name}_fine_tuning.csv"; best_file=out/f"{name}_best.joblib"; oof_file=RUN/f"oof_fine_{name}.npy"
        if result.exists() and best_file.exists() and oof_file.exists():
            rs=pd.read_csv(result); p=np.load(oof_file); allrows.append({"model":name,"trials":len(rs),"valid_trials":int(rs.mean_test_score.notna().sum()),"best_cv_pr_auc":float(np.nanmax(rs.mean_test_score)),**score(y,p)})
    if len(allrows)==3:
        pd.DataFrame(allrows).sort_values("best_cv_pr_auc",ascending=False).to_csv(ROOT/"artifacts"/"top3_fine_tuning_summary.csv",index=False,encoding="utf-8-sig"); mark("top3_selection",top3=top); mark("top3_fine_tuning")

def seed_stability():
    if done("seed_stability"): return
    X,y=load(); cv=folds(y); top=manifest()["top3"]; out=RUN/"seed_stability"; out.mkdir(exist_ok=True); only=os.environ.get("FULL_MODEL")
    for name in top:
        if only and name!=only: continue
        path=out/f"{name}_seeds.csv"; existing=pd.read_csv(path) if path.exists() else pd.DataFrame(); rows=[]
        for seed in [17,42,77,123,2026]:
            if not existing.empty and seed in existing.seed.tolist(): continue
            pipe=clone(joblib.load(RUN/"fine_tuning"/f"{name}_best.joblib"))
            params=pipe.get_params(); key="model__random_seed" if "model__random_seed" in params else "model__random_state"
            pipe.set_params(**{key:seed}); t=time.time(); p=oof(X,y,pipe,cv); rows.append({"model":name,"seed":seed,"seconds":time.time()-t,**score(y,p)})
            pd.concat([existing,pd.DataFrame(rows)],ignore_index=True).to_csv(path,index=False)
    complete=[]
    for name in top:
        path=out/f"{name}_seeds.csv"
        if path.exists() and len(pd.read_csv(path))==5: complete.append(pd.read_csv(path))
    if len(complete)==3:
        full=pd.concat(complete); full.to_csv(RUN/"seed_stability_all.csv",index=False); full.groupby("model")[["pr_auc","roc_auc","f1","recall","precision","fn","fp","seconds"]].agg(["mean","std","min","max"]).to_csv(ROOT/"artifacts"/"seed_stability_summary.csv"); mark("seed_stability")

def main():
    a=argparse.ArgumentParser(); a.add_argument("--stage",default="all",choices=["setup","features","baseline","tree_features","search","fine","seeds","all"]); a.add_argument("--fresh-run",action="store_true",help="archive generated run state under tmp/ and restart every checkpoint"); z=a.parse_args(); RUN.mkdir(parents=True,exist_ok=True); CAND.mkdir(parents=True,exist_ok=True)
    if z.fresh_run:
        archive=prepare_fresh_run()
        print(json.dumps({"fresh_run":True,"archived_previous_state":str(archive.relative_to(ROOT)).replace("\\","/")}))
    actions={"setup":[setup],"features":[setup,feature_experiments],"baseline":[setup,feature_experiments,baseline],"tree_features":[setup,feature_experiments,baseline,tree_feature_validation],"search":[setup,feature_experiments,baseline,tree_feature_validation,randomized],"fine":[setup,feature_experiments,baseline,tree_feature_validation,randomized,fine_tune],"seeds":[seed_stability],"all":[setup,feature_experiments,baseline,tree_feature_validation,randomized,fine_tune,seed_stability]}
    for fn in actions[z.stage]:
        try: fn()
        except Exception as e:
            mark(fn.__name__,"FAILED",blocked_reason=f"{type(e).__name__}: {e}"); (RUN/"failure.log").write_text(traceback.format_exc(),encoding="utf-8"); raise
if __name__=="__main__": main()

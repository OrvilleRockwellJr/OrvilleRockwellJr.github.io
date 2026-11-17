import json, sys, time, numpy as np, torch, pandas as pd
sys.path.insert(0,"/home/claude/build/scripts"); import gd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
torch.set_num_threads(2); torch.manual_seed(0); np.random.seed(0)
D="/home/claude/build/data/"; O="/home/claude/build/models/"; NT="ACGT"
P=json.load(open(D+"dna_prep2.json")); C=pd.DataFrame(P["crispron"]); A=pd.DataFrame(P["azimuth"])
def oh(s):
    a=np.zeros((30,4),np.float32)
    for i,c in enumerate(s): a[i,NT.index(c)]=1
    return a
def x0_of(ss): return torch.tensor(np.stack([oh(s).reshape(-1) for s in ss])*2-1,dtype=torch.float32)
def feats(ss):
    F=[]
    for s in ss:
        a=oh(s).reshape(-1); di=np.zeros((29,16),np.float32)
        for i in range(29): di[i,NT.index(s[i])*4+NT.index(s[i+1])]=1
        gc=sum(c in "GC" for c in s[4:24])/20.0
        F.append(np.concatenate([a,di.reshape(-1),[gc,gc*gc]]))
    return np.array(F)
log=lambda m:print(m,flush=True)
K=C[C.lab=="kim"]; L=C[C.lab=="luo"]; At=A[A.split=="test"]
ktr,kva,kte=[K[K.split==s] for s in ("train","val","test")]
ltr,lva,lte=[L[L.split==s] for s in ("train","val","test")]
# ---- critic: lab A (Kim 2019 library) only
Xtr,Xva=x0_of(ktr.s.tolist()),x0_of(kva.s.tolist())
heads=gd.train_critic_ensemble(Xtr,torch.tensor(ktr.y.values,dtype=torch.float32),Xva,torch.tensor(kva.y.values,dtype=torch.float32),K=5,epochs=40,H=128,wd=1e-2,log=log)
torch.save([h.state_dict() for h in heads],O+"dna_critics.pt")
def critic_scores(ss,t_idx=0):
    X=x0_of(ss); tt=torch.full((len(X),),t_idx,dtype=torch.long); torch.manual_seed(3); xt,_=gd.q_sample(X,tt)
    with torch.no_grad(): return gd.ens_logits(heads,xt,tt).mean(0).numpy()
# ---- judge: lab B (Luo 2020 library) only, different features and model class
best=(-1,None)
Ftr,Fva=feats(ltr.s.tolist()),feats(lva.s.tolist())
for Cc in (0.003,0.01,0.03,0.1,0.3,1):
    lr=LogisticRegression(C=Cc,max_iter=3000).fit(Ftr,ltr.y.values); a=roc_auc_score(lva.y.values,lr.decision_function(Fva))
    if a>best[0]: best=(a,Cc,lr)
lr=best[2]; log(f"judge C={best[1]} val AUC {best[0]:.3f}")
J=lambda ss: lr.decision_function(feats(ss))
res={}
def ev(name,d,fn,ycol="y",cont=None):
    s=fn(d.s.tolist()); r={"n":int(len(d)),"auc":float(roc_auc_score(d[ycol].values,s))}
    if cont: r["spearman"]=float(spearmanr(s,d[cont].values).correlation)
    res[name]=r; log(f"{name}: {r}")
for nm,d in (("kim_test",kte),("luo_test",lte),("azimuth_test",At)):
    cont="eff" if nm!="azimuth_test" else "rank"
    ev("critic@"+nm,d,critic_scores,cont=cont); ev("judge@"+nm,d,J,cont=cont)
res["critic_trained_on"]="Kim 2019 library (n=%d)"%len(ktr); res["judge_trained_on"]="Luo 2020 library (n=%d)"%len(ltr)
# agreement of critic vs judge on random prior sites (what the audit sees)
pr=json.load(open(D+"dna_prep.json"))["prior"][:3000]
res["critic_judge_spearman_prior"]=float(spearmanr(critic_scores(pr),J(pr)).correlation); log(res["critic_judge_spearman_prior"])
json.dump({"coef":lr.coef_[0].tolist(),"b":float(lr.intercept_[0]),"C":best[1],"val_auc":best[0],"test_auc":res["judge@luo_test"]["auc"],"eval":res},open(O+"dna_judge.json","w"))
json.dump(res,open(O+"dna_eval2.json","w"),indent=1)

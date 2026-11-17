import json, sys, time, re, numpy as np, torch
sys.path.insert(0,"/home/claude/build/scripts"); import gd
from train_ae import AE, mk, vocab, L, Z, detok
from rdkit import Chem, RDLogger; from rdkit.Chem import rdFingerprintGenerator as G
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
RDLogger.DisableLog('rdApp.*'); torch.set_num_threads(2); torch.manual_seed(0); np.random.seed(0)
D="/home/claude/build/data/"; O="/home/claude/build/models/"; log=lambda m:print(m,flush=True)
M=json.load(open(D+"mol.json")); ae=AE(); ae.load_state_dict(torch.load(O+"ae.pt")); ae.eval()
@torch.no_grad()
def encode(ss,bs=1024):
    out=[]
    for i in range(0,len(ss),bs): out.append(ae.encode(mk(ss[i:i+bs]))[0])
    return torch.cat(out)
t0=time.time()
Zall=encode(M["ae_train"]); mean=Zall.mean(0); std=Zall.std(0)+1e-6
log(f"encoded pool {Zall.shape}; latent std range {std.min():.3f}-{std.max():.3f}; active dims(std>0.1)={int((std>0.1).sum())}")
torch.save({"mean":mean,"std":std},O+"mol_stats.pt")
Xall=(Zall-mean)/std
# AE quality on held-out ZINC: exact reconstruction (greedy) and token accuracy
va=M["ae_val"][:1000]
with torch.no_grad():
    zv=encode(va); dec=ae.decode(zv); rec=[detok(r) for r in dec]
ex=np.mean([a==b for a,b in zip(rec,va)]); val=np.mean([Chem.MolFromSmiles(r) is not None for r in rec])
log(f"AE held-out ZINC: exact reconstruction {ex:.3f}; reconstructions valid {val:.3f}")
import os
if os.path.exists(O+"mol_denoiser.pt"): log("denoiser cached")
else:
    den=gd.train_denoiser(Xall,steps=int(sys.argv[1]),bs=256,lr=1e-3,H=512,log=log)
    torch.save(den.state_dict(),O+"mol_denoiser.pt"); log(f"mol denoiser done {time.time()-t0:.0f}s")
# critics
gen=G.GetMorganGenerator(radius=2,fpSize=1024)
def fp(s):
    m=Chem.MolFromSmiles(s); return np.array(list(gen.GetFingerprint(m)),dtype=np.float32)
res={}; critics={}; judges={}
for prop in ("bbb","act","tox"):
    S={sp:{k:v for k,v in d.items() if Chem.MolFromSmiles(k) is not None} for sp,d in M["sets"][prop].items()}; data={}
    for sp in ("train","val","test"):
        ss=list(S[sp].keys()); y=torch.tensor([S[sp][s] for s in ss],dtype=torch.float32)
        data[sp]=(ss,(encode(ss)-mean)/std,y)
    heads=gd.train_critic_ensemble(data["train"][1],data["train"][2],data["val"][1],data["val"][2],K=5,epochs=60,H=128,wd=1e-2,seed=1,log=log)
    critics[prop]=heads
    r={"n_train":len(data["train"][0]),"n_test":len(data["test"][0]),"pos_rate_test":float(data["test"][2].mean())}
    for ti in (0,50,100,150):
        X,y=data["test"][1],data["test"][2]; tt=torch.full((len(X),),ti,dtype=torch.long); torch.manual_seed(3); xt,_=gd.q_sample(X,tt)
        with torch.no_grad(): m=gd.ens_logits(heads,xt,tt).mean(0)
        r[f"critic_auc_t{ti}"]=float(roc_auc_score(y.numpy(),m.numpy()))
    # held-out judge: ECFP4 logistic regression (different features + model class)
    F={sp:np.stack([fp(s) for s in data[sp][0]]) for sp in data}
    best=(-1,None,None)
    for C in (0.01,0.03,0.1,0.3,1,3):
        lr=LogisticRegression(C=C,max_iter=4000).fit(F["train"],data["train"][2].numpy()); a=roc_auc_score(data["val"][2].numpy(),lr.decision_function(F["val"]))
        if a>best[0]: best=(a,C,lr)
    lr=best[2]; r["judge_C"]=best[1]; r["judge_auc_test"]=float(roc_auc_score(data["test"][2].numpy(),lr.decision_function(F["test"])))
    judges[prop]={"coef":lr.coef_[0].tolist(),"b":float(lr.intercept_[0])}
    # critic-vs-judge agreement on the test set
    with torch.no_grad(): cm=gd.ens_logits(heads,data["test"][1],torch.zeros(len(data["test"][1]),dtype=torch.long)).mean(0).numpy()
    from scipy.stats import spearmanr
    r["critic_judge_spearman_test"]=float(spearmanr(cm,lr.decision_function(F["test"]))[0])
    res[prop]=r; log(f"{prop}: {json.dumps(r)}")
    if prop=="act":
        ps=M["ptps"]; ss=list(ps.keys()); y=np.array([1 if ps[s]>=5.5 else 0 for s in ss])
        zx=(encode(ss)-mean)/std; tt=torch.zeros(len(ss),dtype=torch.long)
        with torch.no_grad(): cm=gd.ens_logits(heads,zx,tt).mean(0).numpy()
        Fp=np.stack([fp(s) for s in ss]); jm=lr.decision_function(Fp)
        ptp_in_train=sum(s in S["train"] for s in ss)
        def boot(sc):
            rng=np.random.default_rng(0); v=[]
            for _ in range(2000):
                i=rng.integers(0,len(y),len(y))
                if y[i].min()==y[i].max(): continue
                v.append(roc_auc_score(y[i],sc[i]))
            return float(np.percentile(v,2.5)),float(np.percentile(v,97.5))
        res["ptpsigma_external"]={"n":len(ss),"n_active":int(y.sum()),"n_also_in_family_train":int(ptp_in_train),
            "critic_auc":float(roc_auc_score(y,cm)),"critic_ci":boot(cm),"judge_auc":float(roc_auc_score(y,jm)),"judge_ci":boot(jm)}
        log("ptpsigma: "+json.dumps(res["ptpsigma_external"]))
torch.save({p:[h.state_dict() for h in hs] for p,hs in critics.items()},O+"mol_critics.pt")
json.dump(judges,open(O+"mol_judges.json","w")); json.dump(res,open(O+"mol_train_metrics.json","w"),indent=1)
# reference set for nearest-neighbour novelty in the browser (ZINC subsample + labelled actives)
import random; random.seed(0)
ref=random.sample(M["zinc_all"],2500)+[s for s,v in M["sets"]["act"]["train"].items() if v==1]+[s for s,v in M["sets"]["bbb"]["train"].items() if v==1][:1000]
ref=list(dict.fromkeys(ref)); json.dump(ref,open(O+"mol_ref.json","w")); log(f"ref {len(ref)}; total {time.time()-t0:.0f}s")

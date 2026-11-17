import json, sys, time, numpy as np, torch, pandas as pd
sys.path.insert(0,"/home/claude/build/scripts"); import gd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
torch.set_num_threads(1); torch.manual_seed(0); np.random.seed(0)
D="/home/claude/build/data/"; O="/home/claude/build/models/"
P=json.load(open(D+"dna_prep.json")); NT="ACGT"
def oh(s): 
    a=np.zeros((30,4),np.float32)
    for i,c in enumerate(s): a[i,NT.index(c)]=1
    return a
def x0_of(ss): return torch.tensor(np.stack([oh(s).reshape(-1) for s in ss])*2-1,dtype=torch.float32)
prior=P["prior"]; print("prior",len(prior))
log=lambda m:print(m,flush=True)
t0=time.time()
den=gd.train_denoiser(x0_of(prior),steps=int(sys.argv[1]),bs=256,lr=1e-3,H=384,log=log)
torch.save(den.state_dict(),O+"dna_denoiser.pt"); log(f"denoiser done {time.time()-t0:.0f}s")
pairs=pd.DataFrame(P["pairs"])
tr=pairs[pairs.split=="train"]; va=pairs[pairs.split=="val"]; te=pairs[pairs.split=="test"]
Xtr,Xva,Xte=[x0_of(d.s.tolist()) for d in (tr,va,te)]; ytr,yva,yte=[torch.tensor(d.y.values,dtype=torch.float32) for d in (tr,va,te)]
heads=gd.train_critic_ensemble(Xtr,ytr,Xva,yva,K=5,epochs=40,H=128,wd=1e-2,log=log)
torch.save([h.state_dict() for h in heads],O+"dna_critics.pt")
# critic AUC on held-out genes at several noise levels
def auc_at(t_idx):
    out=[]
    for nm,(X,y) in {"val":(Xva,yva),"test":(Xte,yte)}.items():
        tt=torch.full((len(X),),t_idx,dtype=torch.long); torch.manual_seed(3); xt,_=gd.q_sample(X,tt)
        with torch.no_grad(): m=gd.ens_logits(heads,xt,tt).mean(0)
        out.append(roc_auc_score(y.numpy(),m.numpy()))
    return out
for ti in (0,50,100,150): log(f"critic ensemble AUC @t={ti} (ab={gd.AB[ti]:.3f}) val/test: {auc_at(ti)}")
# held-out judge: logistic regression on positional one-hot + dinucleotides + GC (different feature family & model class)
def feats(ss):
    F=[]
    for s in ss:
        a=oh(s).reshape(-1); di=np.zeros((29,16),np.float32)
        for i in range(29): di[i,NT.index(s[i])*4+NT.index(s[i+1])]=1
        gc=sum(c in "GC" for c in s[4:24])/20.0
        F.append(np.concatenate([a,di.reshape(-1),[gc,gc*gc]]))
    return np.array(F)
Ftr,Fva,Fte=[feats(d.s.tolist()) for d in (tr,va,te)]
best=(-1,None)
for C in (0.003,0.01,0.03,0.1,0.3,1):
    lr=LogisticRegression(C=C,max_iter=3000).fit(Ftr,tr.y.values); a=roc_auc_score(va.y.values,lr.decision_function(Fva))
    if a>best[0]: best=(a,C,lr)
lr=best[2]; ja=roc_auc_score(te.y.values,lr.decision_function(Fte)); log(f"judge LR C={best[1]} val AUC {best[0]:.3f} test AUC {ja:.3f}")
json.dump({"coef":lr.coef_[0].tolist(),"b":float(lr.intercept_[0]),"C":best[1],"val_auc":best[0],"test_auc":ja},open(O+"dna_judge.json","w"))
print("DONE",time.time()-t0)

import json,numpy as np
from rdkit import Chem, DataStructs, RDLogger; from rdkit.Chem import rdFingerprintGenerator as G; from rdkit.ML.Cluster import Butina
from sklearn.metrics import roc_auc_score
RDLogger.DisableLog('rdApp.*')
rows=json.load(open("data/ptprs_all_raw.json")); best={}
for r in rows:
    s=r.get("canonical_smiles")
    if not s or Chem.MolFromSmiles(s) is None: continue
    t=r["standard_type"]; v=r["standard_value"]
    if v is None: continue
    v=float(v)
    if t=="IC50" and r["standard_units"]=="nM": lab=1 if v<=3162 else 0
    elif t=="Inhibition" and r["standard_units"]=="%": lab=1 if v>=50 else 0
    else: continue
    k=Chem.MolToSmiles(Chem.MolFromSmiles(s)); best[k]=max(best.get(k,0),lab)
ss=list(best); y=np.array([best[s] for s in ss]); print(len(ss),"mols",y.sum(),"active")
gen=G.GetMorganGenerator(radius=2,fpSize=1024); fps=[gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in ss]
n=len(ss); S=np.array([DataStructs.BulkTanimotoSimilarity(f,fps) for f in fps])
# clusters at Tanimoto 0.4 (series)
d=[]; 
for i in range(1,n):
    for j in range(i): d.append(1-S[i,j])
cl=Butina.ClusterData(d,n,0.6,isDistData=True); g=np.zeros(n,int)
for ci,c in enumerate(cl):
    for m in c: g[m]=ci
print("clusters",len(cl))
def score(tr,te,mode):
    a=[i for i in tr if y[i]==1]; b=[i for i in tr if y[i]==0]
    out=[]
    for i in te:
        sa=S[i,a].max() if a else 0; sb=S[i,b].max() if b else 0
        k=3; ka=np.sort(S[i,a])[-k:].mean() if a else 0; kb=np.sort(S[i,b])[-k:].mean() if b else 0
        out.append({"maxA":sa,"diff":sa-sb,"k3":ka-kb}[mode])
    return np.array(out)
rng=np.random.default_rng(0)
for mode in ("maxA","diff","k3"):
    pred=np.zeros(n)
    for c in set(g):
        te=[i for i in range(n) if g[i]==c]; tr=[i for i in range(n) if g[i]!=c]
        pred[te]=score(tr,te,mode)
    auc=roc_auc_score(y,pred); bs=[]
    for _ in range(2000):
        ix=rng.integers(0,n,n)
        if y[ix].min()==y[ix].max(): continue
        bs.append(roc_auc_score(y[ix],pred[ix]))
    print(mode,"leave-one-series-out AUC %.2f (%.2f-%.2f)"%(auc,*np.percentile(bs,[2.5,97.5])))

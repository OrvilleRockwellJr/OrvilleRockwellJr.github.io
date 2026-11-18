import json,sys,time,re,numpy as np,torch
sys.path.insert(0,"/home/claude/build/scripts"); import gd
from train_ae import AE, mk, vocab, L, Z
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.metrics import roc_auc_score
RDLogger.DisableLog('rdApp.*'); torch.set_num_threads(4); np.random.seed(0)
D="/home/claude/build/data/"; O="/home/claude/build/models/"
log=lambda m:print(m,flush=True)
TOK=re.compile(r"(\[[^\]]+]|Br|Cl|%\d\d|[A-Za-z]|\d|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$)")
unc=rdMolStandardize.Uncharger(); lf=rdMolStandardize.LargestFragmentChooser()
def std(smi):
    m=Chem.MolFromSmiles(str(smi))
    if m is None: return None
    try:
        m=lf.choose(m); m=unc.uncharge(m); Chem.RemoveStereochemistry(m)
        return Chem.MolToSmiles(m,isomericSmiles=False,canonical=True)
    except Exception: return None
M=json.load(open(D+"mol2.json")); keep=set(M["vocab"]); MAXT=M["maxt"]
ok=lambda s: bool(s) and len(TOK.findall(s))<=MAXT and all(t in keep for t in TOK.findall(s))
ae=AE(); ae.load_state_dict(torch.load(O+"ae.pt")); ae.eval()
st=torch.load(O+"mol_stats.pt"); mean,sd=st["mean"],st["std"]
@torch.no_grad()
def enc(ss,bs=512):
    out=[]
    for i in range(0,len(ss),bs): out.append(ae.encode(mk(ss[i:i+bs]))[0])
    return (torch.cat(out)-mean)/sd
# ---------- family data with target provenance
ch=json.load(open(D+"chembl_ptp.json")); by={}
for r in ch:
    s=std(r["smiles"])
    if not ok(s) or not r["nM"] or r["nM"]<=0: continue
    by.setdefault((s,r["target"]),[]).append(9-np.log10(r["nM"]))
tgt={}
for (s,t),ps in by.items(): tgt.setdefault(s,{})[t]=float(np.median(ps))
RECEPTOR={"PTPRS","PTPRS_brain","PTPRF","PTPRB","PTPRC","PTPRG"}
TYPE2A={"PTPRS","PTPRS_brain","PTPRF"}
fam={}
for s,d in tgt.items():
    b=max(d.values())
    if b>=5.5: fam[s]=1
    elif b<5.0: fam[s]=0
log(f"family labelled {len(fam)} active {sum(fam.values())}")
# ---------- eval: all ChEMBL PTPsigma rows (IC50 + % inhibition), held out entirely
rows=json.load(open(D+"ptprs_all_raw.json")); ev={}
for r in rows:
    s=std(r.get("canonical_smiles")); v=r.get("standard_value"); t=r["standard_type"]
    if not s or v is None: continue
    v=float(v)
    if t=="IC50" and r.get("standard_units")=="nM": lab=1 if v<=3162 else 0
    elif t=="Inhibition" and r.get("standard_units")=="%": lab=1 if v>=50 else 0
    else: continue
    ev[s]=max(ev.get(s,0),lab)
ev_enc={s:v for s,v in ev.items() if ok(s)}
log(f"PTPsigma eval: {len(ev)} molecules, {len(ev_enc)} encodable, {sum(ev_enc.values())} active")
# remove eval molecules from every training split
train_pool={s:v for s,v in fam.items() if s not in ev_enc}
splits=M["sets"]["act"]
tr={s:v for s,v in train_pool.items() if s in splits["train"]}
va={s:v for s,v in train_pool.items() if s in splits["val"]}
te={s:v for s,v in train_pool.items() if s in splits["test"]}
leftover={s:v for s,v in train_pool.items() if s not in splits["train"] and s not in splits["val"] and s not in splits["test"]}
tr.update(leftover)
log(f"train {len(tr)} val {len(va)} test {len(te)} (eval molecules removed: {len(fam)-len(train_pool)})")
def recep(s): return any(k in RECEPTOR for k in tgt.get(s,{}))
def t2a(s): return any(k in TYPE2A for k in tgt.get(s,{}))
log(f"receptor-labelled in train {sum(recep(s) for s in tr)}; type-IIa {sum(t2a(s) for s in tr)}")
TR=list(tr); VA=list(va); TE=list(te); EV=list(ev_enc)
Xtr,Xva,Xte,Xev=enc(TR),enc(VA),enc(TE),enc(EV)
ytr=torch.tensor([tr[s] for s in TR],dtype=torch.float32)
yva=torch.tensor([va[s] for s in VA],dtype=torch.float32)
yte=np.array([te[s] for s in TE]); yev=np.array([ev_enc[s] for s in EV])
def boot(y,sc,n=4000):
    rng=np.random.default_rng(0); v=[]
    for _ in range(n):
        i=rng.integers(0,len(y),len(y))
        if y[i].min()==y[i].max(): continue
        v.append(roc_auc_score(y[i],sc[i]))
    return float(np.percentile(v,2.5)),float(np.percentile(v,97.5))
def score(heads,X):
    t=torch.zeros(len(X),dtype=torch.long)
    with torch.no_grad(): return gd.ens_logits(heads,X,t).mean(0).numpy()
ARMS={"uniform":("all",1),"recep_w4":("all",4),"recep_w10":("all",10),"recep_only":("recep",1),"t2a_w10":("t2a",10)}
res={"eval_n":len(EV),"eval_active":int(yev.sum()),"n_train":len(TR),"arms":{}}
for name,(mode,W) in ARMS.items():
    if mode=="recep": idx=[i for i,s in enumerate(TR) if recep(s)]
    else: idx=list(range(len(TR)))
    rep=list(idx)
    if W>1:
        key=t2a if mode=="t2a" else recep
        rep=rep+[i for i in idx if key(TR[i])]*(W-1)
    X=Xtr[rep]; Y=ytr[rep]
    aucs_ev=[]; aucs_te=[]
    for seed in (1,2,3):
        t0=time.time()
        heads=gd.train_critic_ensemble(X,Y,Xva,yva,K=5,epochs=60,H=128,wd=1e-2,seed=seed)
        se=score(heads,Xev); stt=score(heads,Xte)
        aucs_ev.append(roc_auc_score(yev,se)); aucs_te.append(roc_auc_score(yte,stt))
        log(f"{name} seed{seed} n={len(X)} evalAUC {aucs_ev[-1]:.3f} familyTestAUC {aucs_te[-1]:.3f} ({time.time()-t0:.0f}s)")
        if seed==1: last=se
    res["arms"][name]={"n_rows":len(X),"eval_auc_mean":float(np.mean(aucs_ev)),"eval_auc_sd":float(np.std(aucs_ev)),
        "eval_aucs":[float(a) for a in aucs_ev],"family_test_auc_mean":float(np.mean(aucs_te)),
        "eval_ci_seed1":boot(yev,last)}
    log(f"== {name}: eval {np.mean(aucs_ev):.3f}+-{np.std(aucs_ev):.3f}  family {np.mean(aucs_te):.3f}  CI(seed1) {res['arms'][name]['eval_ci_seed1']}")
json.dump(res,open("/home/claude/build/results_act2.json","w"),indent=1)
log("DONE")

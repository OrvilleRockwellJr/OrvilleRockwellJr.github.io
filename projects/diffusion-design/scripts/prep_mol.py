import re, json, gzip, random, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*')
random.seed(0); np.random.seed(0)
D="/home/claude/build/data/"
TOK=re.compile(r"(\[[^\]]+]|Br|Cl|%\d\d|[A-Za-z]|\d|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$)")
unc=rdMolStandardize.Uncharger(); lf=rdMolStandardize.LargestFragmentChooser()
def std(smi):
    m=Chem.MolFromSmiles(smi)
    if m is None: return None
    try:
        m=lf.choose(m); m=unc.uncharge(m)
        Chem.RemoveStereochemistry(m)
        s=Chem.MolToSmiles(m,isomericSmiles=False,canonical=True)
    except Exception: return None
    return s
def toks(s): return TOK.findall(s)
MAXT=46
def scaf(s):
    try: return MurckoScaffold.MurckoScaffoldSmiles(smiles=s)
    except Exception: return ""
# ---- ZINC
z=pd.read_csv(D+"zinc250k.csv"); z=z.smiles.str.strip().tolist()
zs=[]
for s in z:
    t=std(s)
    if t and len(toks(t))<=MAXT: zs.append(t)
print("zinc kept",len(zs),"of",len(z))
# ---- BBB
b=pd.read_csv(D+"b3db.tsv",sep="\t")
bbb={}
for s,l in zip(b.SMILES,b["BBB+/BBB-"]):
    t=std(str(s))
    if t and len(toks(t))<=MAXT: bbb[t]=1 if l=="BBB+" else 0
print("bbb",len(bbb),sum(bbb.values()))
# ---- Tox21 (any of 12 assays active => toxic-flag)
t21=pd.read_csv(D+"tox21.csv.gz"); cols=[c for c in t21.columns if c not in("mol_id","smiles")]
tox={}
for _,r in t21.iterrows():
    s=std(r.smiles)
    if not s or len(toks(s))>MAXT: continue
    v=r[cols].dropna().values
    if len(v)<4: continue
    tox[s]=int(v.max()>=1)
print("tox21",len(tox),sum(tox.values()))
# ---- PTP activity
ch=json.load(open(D+"chembl_ptp.json"))
by={}
for r in ch:
    s=std(r["smiles"])
    if not s or len(toks(s))>MAXT: continue
    p=9-np.log10(r["nM"]) if r["nM"]>0 else None
    if p is None: continue
    by.setdefault((s,r["target"]),[]).append(p)
act={}; tgt={}
for (s,t),ps in by.items():
    p=float(np.median(ps)); tgt.setdefault(s,{})[t]=p
for s,d in tgt.items():
    best=max(d.values())
    if best>=5.5: act[s]=1
    elif best<5.0: act[s]=0
print("ptp-family",len(act),sum(act.values()))
ptps=[s for s,d in tgt.items() if any(k.startswith("PTPRS") for k in d)]
ptps={s:max(v for k,v in tgt[s].items() if k.startswith("PTPRS")) for s in ptps}
print("PTPsigma molecules",len(ptps),"active>=5.5:",sum(v>=5.5 for v in ptps.values()))
# ---- vocab from everything
allsm=set(zs)|set(bbb)|set(tox)|set(act)|set(ptps)
from collections import Counter
c=Counter(t for s in allsm for t in toks(s))
vocab=["<pad>","<bos>","<eos>"]+[t for t,n in c.most_common() if n>=20]
keep=set(vocab)
def ok(s): return all(t in keep for t in toks(s))
zs=[s for s in zs if ok(s)]; 
bbb={s:v for s,v in bbb.items() if ok(s)}; tox={s:v for s,v in tox.items() if ok(s)}
act={s:v for s,v in act.items() if ok(s)}; ptps={s:v for s,v in ptps.items() if ok(s)}
print("vocab",len(vocab),vocab)
# scaffold split for each labelled set
def split(dd):
    sc={}
    for s in dd: sc.setdefault(scaf(s),[]).append(s)
    groups=sorted(sc.values(),key=lambda g:(-len(g),g[0]))
    # big groups to train, rest random
    random.shuffle(groups)
    tr,va,te=[],[],[]
    n=len(dd)
    for g in sorted(groups,key=len,reverse=True):
        if len(tr)+len(g)<=0.8*n: tr+=g
        elif len(va)<0.1*n: va+=g
        else: te+=g
    return {"train":{s:dd[s] for s in tr},"val":{s:dd[s] for s in va},"test":{s:dd[s] for s in te}}
sets={"bbb":split(bbb),"tox":split(tox),"act":split(act)}
for k,v in sets.items(): print(k,{a:(len(b),sum(b.values())) for a,b in v.items()})
# held out judge domain: PTPsigma external set
random.shuffle(zs)
zinc_train=zs[:140000]; zinc_val=zs[140000:142000]
# AE training pool: zinc subset + all labelled molecules (no labels used) -> encoder sees them
pool=list(set(zinc_train)|set(bbb)|set(tox)|set(act)|set(ptps))
random.shuffle(pool)
json.dump({"vocab":vocab,"maxt":MAXT,"ae_train":pool,"ae_val":zinc_val,"zinc_all":zs,"sets":sets,"ptps":ptps},open(D+"mol.json","w"))
print("ae pool",len(pool))

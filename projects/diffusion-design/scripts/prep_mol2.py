import re, json, random, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*'); random.seed(0); np.random.seed(0)
D="/home/claude/build/data/"
TOK=re.compile(r"(\[[^\]]+]|Br|Cl|%\d\d|[A-Za-z]|\d|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$)")
unc=rdMolStandardize.Uncharger(); lf=rdMolStandardize.LargestFragmentChooser()
def std(smi):
    m=Chem.MolFromSmiles(str(smi))
    if m is None: return None
    try:
        m=lf.choose(m); m=unc.uncharge(m); Chem.RemoveStereochemistry(m); return Chem.MolToSmiles(m,isomericSmiles=False,canonical=True)
    except Exception: return None
M=json.load(open(D+"mol.json")); keep=set(M["vocab"]); MAXT=M["maxt"]
ok=lambda s: s and len(TOK.findall(s))<=MAXT and all(t in keep for t in TOK.findall(s))
def scaf(s):
    try: return MurckoScaffold.MurckoScaffoldSmiles(smiles=s)
    except Exception: return ""
def split(dd,fr=(0.8,0.1)):
    sc={}
    for s in dd: sc.setdefault(scaf(s),[]).append(s)
    groups=list(sc.values()); random.shuffle(groups); n=len(dd); tr,va,te=[],[],[]
    for g in sorted(groups,key=len,reverse=True):
        if len(tr)+len(g)<=fr[0]*n: tr+=g
        elif len(va)<fr[1]*n: va+=g
        else: te+=g
    return {"train":{s:dd[s] for s in tr},"val":{s:dd[s] for s in va},"test":{s:dd[s] for s in te}}
# ---------- BBB: judge = MoleculeNet BBBP (Martins 2012 curation); critic = B3DB compounds NOT in BBBP (disjoint compounds)
b=pd.read_csv(D+"b3db.tsv",sep="\t"); bp=pd.read_csv(D+"ext/bbbp.csv")
judge={}
for s_,y in zip(bp.smiles,bp.p_np):
    t=std(s_)
    if ok(t): judge[t]=int(y)
crit={}
for s_,l in zip(b.SMILES,b["BBB+/BBB-"]):
    t=std(s_)
    if ok(t) and t not in judge: crit[t]=1 if l=="BBB+" else 0
print("bbb critic set (B3DB minus BBBP)",len(crit),sum(crit.values()),"| judge set (BBBP)",len(judge),sum(judge.values()))
ext={}
# ---------- Tox: judge on ToxCast non-Tox21 assays, different assays (TOX21_* columns dropped), compounds may overlap
t21=pd.read_csv(D+"tox21.csv.gz"); s21={std(s) for s in t21.smiles}
tc=pd.read_csv(D+"ext/toxcast_data.csv.gz"); cols=[c for c in tc.columns if c!="smiles" and not c.startswith("TOX21")]
tox_j={}; frac=[]
for i,r in tc.iterrows():
    t=std(r.smiles)
    if not ok(t): continue
    v=r[cols].dropna().values
    if len(v)<50: continue
    tox_j[t]=float(np.mean(v>=1))
fr=np.array(list(tox_j.values())); thr=float(np.quantile(fr,0.75)); print("toxcast judge compounds",len(tox_j),"active-fraction q75",thr)
tox_j={s:int(v>thr) for s,v in tox_j.items()}
print("tox judge set",len(tox_j),sum(tox_j.values()))
M["sets"]["bbb"]=split(crit)
M["jsets"]={"bbb":split(judge),"tox":split(tox_j)}; M["ext"]={"bbb":ext}
json.dump(M,open(D+"mol2.json","w")); print("saved")

import json, numpy as np, pandas as pd, openpyxl, random
random.seed(0); np.random.seed(0)
D="/home/claude/build/data/"
rows=list(openpyxl.load_workbook(D+"ext/Luo2020_Kim2019.xlsx",read_only=True)["Luo2020_Kim2019"].iter_rows(values_only=True))[1:]
recs=[]
for r in rows:
    ds=r[1]; s=r[7]; y=r[4] if ds.startswith("Luo") else r[8]
    if ds=="Overlap_Luo_Kim2019": continue          # 49 guides measured by both labs: excluded from training/eval to keep sets disjoint
    recs.append(dict(s=s,lab="kim" if ds.startswith("Kim") else "luo",part="test" if ds=="Kim2019_Test" else "train",eff=float(y)))
d=pd.DataFrame(recs)
print(d.groupby(["lab","part"]).size())
# no guide shared between labs
k=set(d[d.lab=="kim"].s); l=set(d[d.lab=="luo"].s); print("kim∩luo guides:",len(k&l))
d=d[~d.s.isin(k&l)]
# binary label = above the median of its own dataset (rank-style, comparable to Azimuth rank>=0.5)
for lab in ("kim","luo"):
    m=d[(d.lab==lab)&(d.part=="train")].eff.median(); d.loc[d.lab==lab,"y"]=(d.loc[d.lab==lab,"eff"]>=m).astype(int); print(lab,"median",m)
# luo: random 80/10/10 split (judge train/val/test); kim train: 90/10 (critic train/val); kim test fixed
idx=d.index.values
d["split"]="train"
r=np.random.rand(len(d))
d.loc[(d.lab=="luo")&(r>=.8),"split"]="val"; d.loc[(d.lab=="luo")&(r>=.9),"split"]="test"
d.loc[(d.lab=="kim")&(d.part=="train")&(r>=.9),"split"]="val"
d.loc[(d.lab=="kim")&(d.part=="test"),"split"]="test"
print(d.groupby(["lab","split"]).agg(n=("y","size"),pos=("y","mean")))
# overlap with the unsupervised prior (report only)
P=json.load(open(D+"dna_prep.json")); pr=set(P["prior"]); print("prior∩crispron:",sum(s in pr for s in d.s))
json.dump({"crispron":d[["s","lab","split","eff","y"]].to_dict("records"),"azimuth":P["pairs"]},open(D+"dna_prep2.json","w"))

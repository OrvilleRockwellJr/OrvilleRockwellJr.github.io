import json, re, random, numpy as np, pandas as pd
random.seed(0); np.random.seed(0)
D="/home/claude/build/data/"
d=pd.read_csv(D+"azimuth.csv"); d=d.rename(columns={"30mer":"s","Target gene":"gene","score_drug_gene_rank":"rank"})
print("rows",len(d),"unique guides",d.s.nunique(), "PAM GG@25-26:",(d.s.str[25:27]=="GG").mean())
d=d[d.s.str[25:27]=="GG"]
g=d.groupby(["s","gene"],as_index=False).agg(rank=("rank","mean"),n=("rank","size"))
print("guide-gene pairs",len(g))
test_genes={"CD13","CD45","NF2","THY1","CD28"}; val_genes={"CD15","CD43"}
g["split"]=np.where(g.gene.isin(test_genes),"test",np.where(g.gene.isin(val_genes),"val","train"))
g["y"]=(g["rank"]>=0.5).astype(int)
print(g.groupby("split").agg(n=("y","size"),pos=("y","mean")))
# prior: PAM-bearing 30-mers from CDS of random human genes (test genes excluded)
dna=json.load(open(D+"dna.json"))
comp=str.maketrans("ACGT","TGCA")
def rc(s): return s.translate(comp)[::-1]
sites=set()
for tid,seq in dna["cds"].items():
    seq=seq.upper()
    for strand in (seq,rc(seq)):
        for i in range(0,len(strand)-29):
            w=strand[i:i+30]
            if w[25:27]=="GG" and set(w)<=set("ACGT"): sites.add(w)
sites=sorted(sites); random.shuffle(sites)
print("prior sites",len(sites))
# exclude any exact overlap with Doench guides from the prior? keep (unsupervised) but report
dset=set(g.s)
ov=sum(s in dset for s in sites); print("prior∩doench",ov)
sites=[s for s in sites if s not in dset]
json.dump({"pairs":g[["s","gene","rank","y","split"]].to_dict("records"),"prior":sites[:120000]},open(D+"dna_prep.json","w"))
print("saved prior",min(len(sites),120000))

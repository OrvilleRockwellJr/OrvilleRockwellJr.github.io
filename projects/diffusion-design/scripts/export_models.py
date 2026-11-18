import json, sys, numpy as np, torch
sys.path.insert(0,"/home/claude/build/scripts"); import gd
from train_ae import AE, vocab, L, Z
from rdkit import Chem, RDLogger; from rdkit.Chem import rdFingerprintGenerator as G
RDLogger.DisableLog('rdApp.*')
O="/home/claude/build/models/"; W="/home/claude/build/web/models/"
import os; os.makedirs(W,exist_ok=True)
def pack(d):
    man={}; chunks=[]; off=0
    for k,v in d.items():
        a=np.ascontiguousarray(np.asarray(v,dtype=np.float32)); man[k]={"off":off,"n":int(a.size),"shape":list(a.shape)}; chunks.append(a.reshape(-1)); off+=a.size
    return man,np.concatenate(chunks)
def den_dict(sd,prefix="den."):
    m={"t1.w":sd["t1.weight"],"t1.b":sd["t1.bias"],"inp.w":sd["inp.weight"],"inp.b":sd["inp.bias"],"out.w":sd["out.weight"],"out.b":sd["out.bias"]}
    i=0
    while f"w1.{i}.weight" in sd:
        m[f"w1.{i}.w"]=sd[f"w1.{i}.weight"]; m[f"w1.{i}.b"]=sd[f"w1.{i}.bias"]; m[f"wt.{i}.w"]=sd[f"wt.{i}.weight"]
        m[f"w2.{i}.w"]=sd[f"w2.{i}.weight"]; m[f"w2.{i}.b"]=sd[f"w2.{i}.bias"]; i+=1
    return {prefix+k:v.numpy() for k,v in m.items()}
def crit_dict(sd,prefix):
    return {prefix+f"{l}.{a}":sd[f"{l}.{b}"].numpy() for l in ("l1","l2","l3") for a,b in (("w","weight"),("b","bias"))}
def build_crit(D,sd,H):
    c=gd.Critic(D,H); c.load_state_dict(sd); c.eval(); return c
# ---- molecules
ae=AE(); ae.load_state_dict(torch.load(O+"ae.pt")); ae.eval(); st=torch.load(O+"mol_stats.pt")
d={}
d.update(den_dict(torch.load(O+"mol_denoiser.pt")))
cr=torch.load(O+"mol_critics.pt")
for prop,sds in cr.items():
    for k,sd in enumerate(sds): d.update(crit_dict(sd,f"{prop}.{k}."))
sdae=ae.state_dict()
d["dec.zh.w"]=sdae["zh.weight"].numpy(); d["dec.zh.b"]=sdae["zh.bias"].numpy(); d["dec.demb"]=sdae["demb.weight"].numpy()
d["dec.gru.wih"]=sdae["gru.weight_ih_l0"].numpy(); d["dec.gru.whh"]=sdae["gru.weight_hh_l0"].numpy(); d["dec.gru.bih"]=sdae["gru.bias_ih_l0"].numpy(); d["dec.gru.bhh"]=sdae["gru.bias_hh_l0"].numpy()
d["dec.out.w"]=sdae["out.weight"].numpy(); d["dec.out.b"]=sdae["out.bias"].numpy()
# encoder (needed in the browser for seeded/analog runs and for scoring a pasted SMILES)
for k,src in (("enc.emb","emb.weight"),("enc.c1.w","c1.weight"),("enc.c1.b","c1.bias"),("enc.c2.w","c2.weight"),("enc.c2.b","c2.bias"),
              ("enc.c3.w","c3.weight"),("enc.c3.b","c3.bias"),("enc.fc.w","fc.weight"),("enc.fc.b","fc.bias"),("enc.mu.w","mu.weight"),("enc.mu.b","mu.bias")):
    d[k]=sdae[src].numpy()
d["stat.mean"]=st["mean"].numpy(); d["stat.std"]=st["std"].numpy()
man,arr=pack(d); arr.tofile(W+"mol.bin"); json.dump({"manifest":man,"vocab":vocab,"L":L,"Z":Z,"T":gd.T},open(W+"mol.json","w"))
print("mol.bin",arr.nbytes/1e6,"MB")
# ---- DNA
d={}; d.update(den_dict(torch.load(O+"dna_denoiser.pt")))
for k,sd in enumerate(torch.load(O+"dna_critics.pt")): d.update(crit_dict(sd,f"eff.{k}."))
man,arr=pack(d); arr.tofile(W+"dna.bin"); json.dump({"manifest":man,"T":gd.T},open(W+"dna.json","w")); print("dna.bin",arr.nbytes/1e6,"MB")
# ---- judges + reference
mj=json.load(open(O+"mol_judges.json")); dj=json.load(open(O+"dna_judge.json"))
json.dump({"mol":mj,"dna":dj},open(W+"judges.json","w"),separators=(",",":"))
ref=json.load(open(O+"mol_ref.json")); gen=G.GetMorganGenerator(radius=2,fpSize=1024)
refo=[]
for s in ref:
    m=Chem.MolFromSmiles(s); refo.append([s,list(gen.GetFingerprint(m).GetOnBits())])
json.dump(refo,open(W+"ref.json","w"),separators=(",",":")); print("ref",len(refo))

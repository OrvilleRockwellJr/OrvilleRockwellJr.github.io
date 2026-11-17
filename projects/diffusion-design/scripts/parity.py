"""Dump PyTorch reference vectors so engine.js can be checked against gd.py."""
import json, sys, numpy as np, torch
sys.path.insert(0,"/home/claude/build/scripts"); import gd
from train_ae import AE, vocab, L, Z, detok
O="/home/claude/build/models/"; W="/home/claude/build/web/models/"
torch.manual_seed(5)
out={}
for name,dsz,H in (("mol",64,128),("dna",120,128)):
    den=gd.Denoiser(dsz,512 if name=="mol" else 384); den.load_state_dict(torch.load(O+f"{name}_denoiser.pt")); den.eval()
    if name=="mol":
        cr=torch.load(O+"mol_critics.pt"); heads_sd=cr["bbb"]; heads_sd2=cr["act"]; heads_sd3=cr["tox"]
        objs=[("bbb",cr["bbb"],1,1.0),("act",cr["act"],1,1.0),("tox",cr["tox"],0,1.0)]
    else:
        objs=[("eff",torch.load(O+"dna_critics.pt"),1,1.0)]
    obj=[]
    for nm,sds,want,w in objs:
        hs=[]
        for sd in sds:
            Hh=sd["l1.weight"].shape[0]; c=gd.Critic(dsz,Hh); c.load_state_dict(sd); c.eval(); hs.append(c)
        obj.append(dict(heads=hs,want=want,weight=w,name=nm))
    x=torch.randn(3,dsz); t=torch.tensor([150,60,5])
    with torch.no_grad(): eps=den(x,t)
    rec={"x":x.tolist(),"t":t.tolist(),"eps":eps.tolist()}
    cg=[]
    for ob in obj:
        m,sd,g=gd.critic_grad(ob["heads"],x,t,ob["want"]); cg.append({"name":ob["name"],"mean":m.tolist(),"sd":sd.tolist(),"grad":g.tolist()})
    rec["crit"]=cg
    noise=torch.randn(2,dsz,generator=torch.Generator().manual_seed(11))
    for gate in ("none","ens"):
        xs=gd.sample(den,dsz,2,steps=20,objectives=obj,gamma=3.0,tau=1.5,eta=0.0,noise=noise,gate_mode=gate,clip=None if name=="mol" else 1.5)
        rec[f"sample_{gate}"]=xs.tolist()
    rec["noise"]=noise.tolist(); out[name]=rec
ae=AE(); ae.load_state_dict(torch.load(O+"ae.pt")); ae.eval()
z=torch.randn(4,Z)*0.8; toks=ae.decode(z)
out["dec"]={"z":z.tolist(),"tokens":toks.tolist(),"smiles":[detok(r) for r in toks]}
json.dump(out,open("/home/claude/build/models/parity.json","w"))
print("parity vectors written")

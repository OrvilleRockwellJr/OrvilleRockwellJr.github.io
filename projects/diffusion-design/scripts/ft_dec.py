import json,sys,time,math,os,numpy as np,torch,torch.nn.functional as F
sys.path.insert(0,"/home/claude/build/scripts")
from train_ae import AE, mk, vocab, V, detok
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
torch.set_num_threads(2); torch.manual_seed(1)
D="/home/claude/build/data/"; O="/home/claude/build/models/"
M=json.load(open(D+"mol2.json")); st=torch.load(O+"mol_stats.pt"); std=st["std"]
ae=AE(); src=O+("ae_dec.pt" if os.path.exists(O+"ae_dec.pt") and "resume" in sys.argv else "ae.pt"); ae.load_state_dict(torch.load(src)); 
for p in ae.parameters(): p.requires_grad=False
dec=[ae.zh,ae.demb,ae.gru,ae.out]
for m in dec:
    for p in m.parameters(): p.requires_grad=True
params=[p for m in dec for p in m.parameters()]
tr=mk(M["ae_train"]); va=mk(M["ae_val"][:600]); print("train",tr.shape,flush=True)
@torch.no_grad()
def evalv(sig,n=600):
    ae.eval(); mu,_=ae.encode(va[:n]); g=torch.Generator().manual_seed(7); z=mu+std*sig*torch.randn(mu.shape,generator=g)
    rec=[detok(r) for r in ae.decode(z)]; ok=np.mean([Chem.MolFromSmiles(r) is not None and len(r)>0 for r in rec]); return ok
def report(tag): print(tag,{s:round(evalv(s),3) for s in (0,0.3,0.6,0.9)},flush=True)
report("before")
opt=torch.optim.Adam(params,lr=1e-3); BUD=float(sys.argv[1])*60; t0=time.time(); bs=256; step=0
while True:
    frac=(time.time()-t0)/BUD
    if frac>=1: break
    for g in opt.param_groups: g["lr"]=1e-3*0.5*(1+math.cos(math.pi*frac))+5e-5
    x=tr[torch.randint(0,len(tr),(bs,))]
    with torch.no_grad(): mu,_=ae.encode(x)
    s=torch.rand(bs,1)*1.0  # noise scale 0..1 of latent std
    z=mu+std*s*torch.randn_like(mu)
    lg=ae.decode_tf(z,x); loss=F.cross_entropy(lg.reshape(-1,V),x.reshape(-1),ignore_index=0)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step(); step+=1
    if step%300==0: print(step,f"{time.time()-t0:.0f}s loss={loss.item():.3f}",flush=True)
torch.save(ae.state_dict(),O+"ae_dec.pt"); report("after")

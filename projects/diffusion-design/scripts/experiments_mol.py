"""Held-out evaluation of classifier-guided latent diffusion for molecules. Settings fixed before looking at results:
   steps=60, DDIM eta=0, N=250 per (setting, seed), seeds 0-2 (identical start noise across settings), tau=1.0, objective weights 1/1/1."""
import json, sys, time, numpy as np, torch, itertools
sys.path.insert(0,"/home/claude/build/scripts"); import gd
from train_ae import AE, vocab, L, Z, detok
from rdkit import Chem, RDLogger; from rdkit.Chem import rdFingerprintGenerator as G, QED, Descriptors, Crippen
RDLogger.DisableLog('rdApp.*'); torch.set_num_threads(2)
D="/home/claude/build/data/"; O="/home/claude/build/models/"; log=lambda m:print(m,flush=True)
M=json.load(open(D+"mol2.json")); st=torch.load(O+"mol_stats.pt"); mean,std=st["mean"],st["std"]
ae=AE(); ae.load_state_dict(torch.load(O+"ae.pt")); ae.eval()
den=gd.Denoiser(64,512); den.load_state_dict(torch.load(O+"mol_denoiser.pt")); den.eval()
cr=torch.load(O+"mol_critics.pt"); heads={}
for p,sds in cr.items():
    hs=[]
    for sd in sds: c=gd.Critic(64,sd["l1.weight"].shape[0]); c.load_state_dict(sd); c.eval(); hs.append(c)
    heads[p]=hs
J=json.load(open(O+"mol_judges.json")); Jc={p:(np.array(v["coef"]),v["b"]) for p,v in J.items()}
gen=G.GetMorganGenerator(radius=2,fpSize=1024)
train_set=set(M["ae_train"])|set(M["zinc_all"])
STEPS=60; N=int(sys.argv[1]) if len(sys.argv)>1 else 250; SEEDS=[0,1,2]; TAU=1.0
def objectives(w=(1,1,1)):
    return [dict(heads=heads["bbb"],want=1,weight=w[0]),dict(heads=heads["act"],want=1,weight=w[1]),dict(heads=heads["tox"],want=0,weight=w[2])]
def critic_scores(x):
    t=torch.zeros(len(x),dtype=torch.long)
    with torch.no_grad(): return {p:gd.ens_logits(heads[p],x,t).mean(0).numpy() for p in heads}
def judge_scores(smis):
    out={p:[] for p in J}
    for s in smis:
        m=Chem.MolFromSmiles(s)
        if m is None:
            for p in J: out[p].append(np.nan)
            continue
        f=np.array(list(gen.GetFingerprint(m)),dtype=np.float32)
        for p,(c,b) in Jc.items(): out[p].append(float(f@c+b))
    return {p:np.array(v) for p,v in out.items()}
sig=lambda z:1/(1+np.exp(-z))
def run(gamma,gate,seed):
    g=torch.Generator().manual_seed(1000+seed); noise=torch.randn(N,64,generator=g)
    x=gd.sample(den,64,N,STEPS,objectives() if gamma>0 else [],gamma=gamma,tau=TAU,eta=0.0,noise=noise,gate_mode=gate)
    with torch.no_grad(): dec=ae.decode(x*std+mean)
    smis=[detok(r) for r in dec]; cs=critic_scores(x); js=judge_scores(smis)
    valid=np.array([Chem.MolFromSmiles(s) is not None and len(s)>0 for s in smis])
    return smis,valid,cs,js,x
def summarize(smis,valid,cs,js):
    n=len(smis); v=valid
    vs=[Chem.CanonSmiles(s) for s,ok in zip(smis,v) if ok]
    uniq=len(set(vs)); novel=len([s for s in set(vs) if s not in train_set])
    r={"n":n,"valid":float(v.mean()),"unique_of_valid":uniq/max(1,len(vs)),"novel_of_unique":novel/max(1,uniq)}
    for p in ("bbb","act","tox"):
        r[f"critic_p_{p}"]=float(np.mean(sig(cs[p])))               # all samples
        r[f"judge_p_{p}"]=float(np.nanmean(sig(js[p])))             # valid only
        r[f"gap_{p}"]=float(np.mean(sig(cs[p])[v])-np.nanmean(sig(js[p])[v]))
    ok=v&(js["bbb"]>0)&(js["act"]>0)&(js["tox"]<0)
    r["judge_success_all3"]=float(ok.mean())
    r["judge_success_bbb_act"]=float((v&(js["bbb"]>0)&(js["act"]>0)).mean())
    ms=[Chem.MolFromSmiles(s) for s,ok_ in zip(smis,v) if ok_]
    r["qed"]=float(np.mean([QED.qed(m) for m in ms])) if ms else 0.0; r["mw"]=float(np.mean([Descriptors.MolWt(m) for m in ms])) if ms else 0.0
    # top-k (10%) by combined critic score: precision under the held-out judge
    comb=cs["bbb"]+cs["act"]-cs["tox"]; comb=np.where(v,comb,-1e9); k=max(1,n//10); top=np.argsort(-comb)[:k]
    r["topk_judge_success_all3"]=float(ok[top].mean()); r["topk_valid"]=float(v[top].mean())
    return r
if __name__=="__main__":
    t0=time.time(); results=[]; examples={}
    settings=[(0.0,"none")]+[(g,gt) for g in (1.0,2.0,4.0,8.0) for gt in ("none","ens")]
    for gamma,gate in settings:
        rs=[]
        for seed in SEEDS:
            smis,valid,cs,js,x=run(gamma,gate,seed); rs.append(summarize(smis,valid,cs,js))
            if seed==0: examples[f"{gamma}_{gate}"]=[s for s,ok in zip(smis,valid) if ok][:12]
        agg={k:[float(np.mean([r[k] for r in rs])),float(np.std([r[k] for r in rs],ddof=1))] for k in rs[0] if k!="n"}
        results.append({"gamma":gamma,"gate":gate,"metrics":agg}); 
        a=agg; log(f"gamma={gamma} gate={gate}: valid {a['valid'][0]:.2f} uniq {a['unique_of_valid'][0]:.2f} novel {a['novel_of_unique'][0]:.2f} | critic p bbb/act/tox {a['critic_p_bbb'][0]:.2f}/{a['critic_p_act'][0]:.2f}/{a['critic_p_tox'][0]:.2f} | judge {a['judge_p_bbb'][0]:.2f}/{a['judge_p_act'][0]:.2f}/{a['judge_p_tox'][0]:.2f} | success3 {a['judge_success_all3'][0]:.3f} topk {a['topk_judge_success_all3'][0]:.3f} ({time.time()-t0:.0f}s)")
    # reference rows: ZINC random molecules and known brain-penetrant set (judge only)
    rng=np.random.default_rng(0); zr=list(rng.choice(M["zinc_all"],1000,replace=False)); js=judge_scores(zr)
    ref={"zinc_random":{"judge_p_bbb":float(np.mean(sig(js["bbb"]))),"judge_p_act":float(np.mean(sig(js["act"]))),"judge_p_tox":float(np.mean(sig(js["tox"]))),
          "judge_success_all3":float(((js["bbb"]>0)&(js["act"]>0)&(js["tox"]<0)).mean())}}
    json.dump({"N":N,"seeds":SEEDS,"steps":STEPS,"tau":TAU,"results":results,"reference":ref,"examples":examples},open(O+"mol_experiments.json","w"),indent=1)
    log("done "+str(time.time()-t0))

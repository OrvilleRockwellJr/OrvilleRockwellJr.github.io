"""Held-out evaluation of classifier-guided diffusion for gene-editing guide sequences (SpCas9, 30-mer = 4 + 20 protospacer + NGG + 3).
   Fixed before looking: steps=50, DDIM eta=0, clip 1.5, N=300 per (setting, seed), seeds 0-2, tau=1.0."""
import json, sys, time, numpy as np, torch, pandas as pd
sys.path.insert(0,"/home/claude/build/scripts"); import gd
torch.set_num_threads(2)
D="/home/claude/build/data/"; O="/home/claude/build/models/"; log=lambda m:print(m,flush=True)
P=json.load(open(D+"dna_prep.json")); NT="ACGT"
den=gd.Denoiser(120,384); den.load_state_dict(torch.load(O+"dna_denoiser.pt")); den.eval()
heads=[]
for sd in torch.load(O+"dna_critics.pt"): c=gd.Critic(120,sd["l1.weight"].shape[0]); c.load_state_dict(sd); c.eval(); heads.append(c)
J=json.load(open(O+"dna_judge.json")); coef=np.array(J["coef"]); jb=J["b"]
def oh(s): 
    a=np.zeros((30,4),np.float32)
    for i,c in enumerate(s): a[i,NT.index(c)]=1
    return a
def feats(s):
    a=oh(s).reshape(-1); di=np.zeros((29,16),np.float32)
    for i in range(29): di[i,NT.index(s[i])*4+NT.index(s[i+1])]=1
    gc=sum(c in "GC" for c in s[4:24])/20.0
    return np.concatenate([a,di.reshape(-1),[gc,gc*gc]])
def jscore(ss): return np.array([feats(s)@coef+jb for s in ss])
def seq_of(x): 
    a=x.reshape(-1,30,4).argmax(-1).numpy(); return ["".join(NT[i] for i in r) for r in a]
sig=lambda z:1/(1+np.exp(-z)); STEPS=50; N=int(sys.argv[1]) if len(sys.argv)>1 else 300; SEEDS=[0,1,2]; TAU=1.0
P2=json.load(open(D+"dna_prep2.json")); pairs=pd.DataFrame(P["pairs"]); known=set(pairs.s)|set(P["prior"])|{r["s"] for r in P2["crispron"]}
def usable(s): 
    gc=sum(c in "GC" for c in s[4:24])/20; return s[25:27]=="GG" and 0.4<=gc<=0.7 and "TTTT" not in s[4:24]
def run(gamma,gate,seed):
    g=torch.Generator().manual_seed(2000+seed); noise=torch.randn(N,120,generator=g)
    x=gd.sample(den,120,N,STEPS,[dict(heads=heads,want=1,weight=1.0)] if gamma>0 else [],gamma=gamma,tau=TAU,eta=0.0,noise=noise,gate_mode=gate,clip=1.5)
    return seq_of(x),x
def critic(x):
    t=torch.zeros(len(x),dtype=torch.long)
    with torch.no_grad(): return gd.ens_logits(heads,x,t).mean(0).numpy()
def summarize(ss,x):
    n=len(ss); pam=np.array([s[25:27]=="GG" for s in ss]); us=np.array([usable(s) for s in ss]); js=jscore(ss); cs=critic(x)
    r={"pam_valid":float(pam.mean()),"unique":len(set(ss))/n,"novel_of_unique":len([s for s in set(ss) if s not in known])/max(1,len(set(ss))),
       "usable":float(us.mean()),"critic_p":float(np.mean(sig(cs))),"judge_p":float(np.mean(sig(js))),"gap":float(np.mean(sig(cs))-np.mean(sig(js))),
       "judge_success":float(((js>0)&us).mean()),"gc_mean":float(np.mean([sum(c in "GC" for c in s[4:24])/20 for s in ss]))}
    comb=np.where(us,cs,-1e9); k=max(1,n//10); top=np.argsort(-comb)[:k]; r["topk_judge_success"]=float(((js>0)&us)[top].mean())
    return r
if __name__=="__main__":
    t0=time.time(); results=[]; ex={}
    for gamma,gate in [(0.0,"none")]+[(g,gt) for g in (1.0,2.0,4.0,8.0,16.0) for gt in ("none","ens")]:
        rs=[]
        for seed in SEEDS:
            ss,x=run(gamma,gate,seed); rs.append(summarize(ss,x))
            if seed==0: ex[f"{gamma}_{gate}"]=ss[:12]
        agg={k:[float(np.mean([r[k] for r in rs])),float(np.std([r[k] for r in rs],ddof=1))] for k in rs[0]}
        results.append({"gamma":gamma,"gate":gate,"metrics":agg}); a=agg
        log(f"gamma={gamma} gate={gate}: pam {a['pam_valid'][0]:.2f} uniq {a['unique'][0]:.2f} novel {a['novel_of_unique'][0]:.2f} usable {a['usable'][0]:.2f} | critic {a['critic_p'][0]:.2f} judge {a['judge_p'][0]:.2f} | success {a['judge_success'][0]:.3f} topk {a['topk_judge_success'][0]:.3f} ({time.time()-t0:.0f}s)")
    te=pairs[pairs.split=="test"]; js=jscore(te.s.tolist()); us=np.array([usable(s) for s in te.s])
    prior=P["prior"][:2000]; jp=jscore(prior); up=np.array([usable(s) for s in prior])
    ref={"doench_test_guides":{"judge_p":float(np.mean(sig(js))),"usable":float(us.mean()),"judge_success":float(((js>0)&us).mean())},
         "genomic_prior_sites":{"judge_p":float(np.mean(sig(jp))),"usable":float(up.mean()),"judge_success":float(((jp>0)&up).mean())}}
    ref["kim_test_guides"]={"judge_p":float(np.mean(sig(jscore([r["s"] for r in P2["crispron"] if r["lab"]=="kim" and r["split"]=="test"])))),"n":526}
    ref["luo_test_guides"]={"judge_p":float(np.mean(sig(jscore([r["s"] for r in P2["crispron"] if r["lab"]=="luo" and r["split"]=="test"])))),"n":1051}
    json.dump({"N":N,"seeds":SEEDS,"steps":STEPS,"tau":TAU,"results":results,"reference":ref,"examples":ex},open(O+"dna_experiments.json","w"),indent=1); log("done "+str(time.time()-t0))

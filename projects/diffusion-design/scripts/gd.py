"""Shared guided-diffusion engine (training + reference sampler). Mirrored 1:1 by engine.js in the browser."""
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
T=200
def alpha_bar_table(T=T,s=0.008):
    t=torch.arange(T+1,dtype=torch.float64)/T
    f=torch.cos((t+s)/(1+s)*math.pi/2)**2
    ab=(f/f[0]).clamp(1e-5,0.9999)
    return ab[1:].float()       # index i -> alpha_bar at step i+1 (i=0..T-1)
AB=alpha_bar_table()
def tfeat(t,dim=32):
    # t: LongTensor (B,) in [0,T-1]; sinusoidal features of t/T
    x=(t.float()/T)[:,None]*1000.0
    fr=torch.exp(-math.log(10000)*torch.arange(dim//2)/(dim//2))[None]
    a=x*fr; return torch.cat([torch.sin(a),torch.cos(a)],1)
class Denoiser(nn.Module):
    def __init__(s,D,H=512,nb=3,td=32):
        super().__init__(); s.D=D; s.td=td
        s.t1=nn.Linear(td,128); s.inp=nn.Linear(D+128,H)
        s.w1=nn.ModuleList([nn.Linear(H,H) for _ in range(nb)]); s.wt=nn.ModuleList([nn.Linear(128,H,bias=False) for _ in range(nb)])
        s.w2=nn.ModuleList([nn.Linear(H,H) for _ in range(nb)]); s.out=nn.Linear(H,D)
    def forward(s,x,t):
        te=F.silu(s.t1(tfeat(t,s.td))); h=s.inp(torch.cat([x,te],1))
        for w1,wt,w2 in zip(s.w1,s.wt,s.w2): h=h+w2(F.silu(w1(h)+wt(te)))
        return s.out(F.silu(h))
class Critic(nn.Module):
    """one head: [x_t, sqrt(ab), sqrt(1-ab)] -> 1 logit"""
    def __init__(s,D,H=256,p=0.1):
        super().__init__(); s.l1=nn.Linear(D+2,H); s.l2=nn.Linear(H,H); s.l3=nn.Linear(H,1); s.dp=nn.Dropout(p)
    def forward(s,x,t):
        ab=AB[t][:,None]; h=torch.cat([x,ab.sqrt(),(1-ab).sqrt()],1)
        h=s.dp(F.silu(s.l1(h))); h=s.dp(F.silu(s.l2(h))); return s.l3(h)[:,0]
def q_sample(x0,t,eps=None):
    if eps is None: eps=torch.randn_like(x0)
    ab=AB[t][:,None]; return ab.sqrt()*x0+(1-ab).sqrt()*eps, eps
def train_denoiser(x0_all,steps,bs=256,lr=1e-3,H=512,log=None,seed=0):
    torch.manual_seed(seed); D=x0_all.shape[1]; m=Denoiser(D,H); opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=0.0)
    ema=Denoiser(D,H); ema.load_state_dict(m.state_dict()); 
    for st in range(steps):
        for g in opt.param_groups: g["lr"]=lr*min(1,(st+1)/300)*(0.5*(1+math.cos(math.pi*st/steps))*0.98+0.02)
        i=torch.randint(0,len(x0_all),(bs,)); x0=x0_all[i]; t=torch.randint(0,T,(bs,))
        xt,eps=q_sample(x0,t); loss=F.mse_loss(m(xt,t),eps)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        with torch.no_grad():
            d=min(0.999,(1+st)/(10+st))
            for pe,pm in zip(ema.parameters(),m.parameters()): pe.mul_(d).add_(pm.detach(),alpha=1-d)
        if log and st%500==0: log(f"  denoiser step {st}/{steps} loss {loss.item():.4f}")
    return ema
def train_critic_ensemble(x0,y,x0v,yv,K=5,epochs=60,bs=128,lr=2e-3,wd=1e-3,H=256,seed=0,log=None):
    """bootstrap ensemble, trained on noised inputs; early-stop on val AUC measured with mid-noise inputs"""
    from sklearn.metrics import roc_auc_score
    D=x0.shape[1]; heads=[]; g=torch.Generator().manual_seed(seed)
    for k in range(K):
        torch.manual_seed(seed*100+k); n=len(x0); idx=torch.randint(0,n,(n,),generator=g)
        X,Y=x0[idx],y[idx]; c=Critic(D,H); opt=torch.optim.AdamW(c.parameters(),lr=lr,weight_decay=wd)
        best=(-1,None)
        for ep in range(epochs):
            c.train(); perm=torch.randperm(n)
            for b in range(0,n,bs):
                j=perm[b:b+bs]; t=torch.randint(0,T,(len(j),)); xt,_=q_sample(X[j],t)
                loss=F.binary_cross_entropy_with_logits(c(xt,t),Y[j]); opt.zero_grad(); loss.backward(); opt.step()
            c.eval()
            with torch.no_grad():
                tt=torch.full((len(x0v),),T//4,dtype=torch.long); torch.manual_seed(7); xv,_=q_sample(x0v,tt)
                a=roc_auc_score(yv.numpy(),c(xv,tt).numpy())
            if a>best[0]: best=(a,{kk:v.clone() for kk,v in c.state_dict().items()})
        c.load_state_dict(best[1]); c.eval(); heads.append(c)
        if log: log(f"  head {k} best val AUC (t=T/4) {best[0]:.3f}")
    return heads
def ens_logits(heads,x,t):
    return torch.stack([h(x,t) for h in heads],0)           # (K,B)
def critic_grad(heads,x,t,want):
    """returns mean logit (B,), std over heads (B,), grad of mean log p(want|x) w.r.t. x (B,D)"""
    x=x.detach().requires_grad_(True); lg=ens_logits(heads,x,t); m=lg.mean(0); sd=lg.std(0)
    sgn=1.0 if want==1 else -1.0
    lp=F.logsigmoid(sgn*m).sum(); g,=torch.autograd.grad(lp,x)
    return m.detach(),sd.detach(),g
@torch.no_grad()
def _eps(den,x,t): return den(x,t)
def sample(den,D,n,steps,objectives,gamma=0.0,tau=None,eta=0.0,seed=0,noise=None,clip=None,gate_mode="none",trust_ramp=False,return_traj=False):
    """objectives: list of dict(heads=..., want=1|0, weight=float). gate_mode: 'none' | 'ens' (ensemble-disagreement gate).
       DDIM with classifier guidance (Dhariwal & Nichol 2021, Alg. 2 form): eps_hat = eps - sqrt(1-ab)*gamma*gate*sum_k w_k grad log p_k(want|x_t)."""
    gen=torch.Generator().manual_seed(seed)
    x=torch.randn(n,D,generator=gen) if noise is None else noise.clone()
    ts=torch.linspace(T-1,0,steps).round().long().tolist(); ts=sorted(set(ts),reverse=True)
    for si,ti in enumerate(ts):
        t=torch.full((n,),ti,dtype=torch.long); ab=AB[ti]; abp=AB[ts[si+1]] if si+1<len(ts) else torch.tensor(1.0)
        with torch.no_grad(): eps=den(x,t)
        if gamma>0 and objectives:
            gsum=torch.zeros_like(x)
            for ob in objectives:
                m,sd,g=critic_grad(ob["heads"],x,t,ob["want"])
                if gate_mode=="ens": gate=(1.0/(1.0+(sd/tau)**2))[:,None]
                else: gate=1.0
                gsum=gsum+ob["weight"]*gate*g
            eps=eps-(1-ab).sqrt()*gamma*gsum
        x0=(x-(1-ab).sqrt()*eps)/ab.sqrt()
        if clip is not None: x0=x0.clamp(-clip,clip); eps=(x-ab.sqrt()*x0)/(1-ab).sqrt()
        sig=eta*((1-abp)/(1-ab)).sqrt()*(1-ab/abp).clamp(min=0).sqrt()
        z=torch.randn(x.shape,generator=gen) if eta>0 else torch.zeros_like(x)
        x=abp.sqrt()*x0+(1-abp-sig**2).clamp(min=0).sqrt()*eps+sig*z
    return x

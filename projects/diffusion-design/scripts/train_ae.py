import json, re, time, math, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(2); torch.manual_seed(0); random.seed(0); np.random.seed(0)
D="/home/claude/build/data/"; O="/home/claude/build/models/"
import os; os.makedirs(O,exist_ok=True)
M=json.load(open(D+"mol.json")); vocab=M["vocab"]; V=len(vocab); L=48; Z=64
idx={t:i for i,t in enumerate(vocab)}
TOK=re.compile(r"(\[[^\]]+]|Br|Cl|%\d\d|[A-Za-z]|\d|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$)")
def enc(s):
    t=[idx[x] for x in TOK.findall(s)]+[2]
    return t+[0]*(L-len(t))
def mk(ss): return torch.tensor([enc(s) for s in ss],dtype=torch.long)
class AE(nn.Module):
    def __init__(s,V=V,E=48,H=256,Z=Z):
        super().__init__()
        s.emb=nn.Embedding(V,E,padding_idx=0)
        s.c1=nn.Conv1d(E,128,5,padding=2); s.c2=nn.Conv1d(128,128,5,stride=2,padding=2); s.c3=nn.Conv1d(128,192,5,stride=2,padding=2)
        s.fc=nn.Linear(192*(L//4),256); s.mu=nn.Linear(256,Z); s.lv=nn.Linear(256,Z)
        s.zh=nn.Linear(Z,H); s.demb=nn.Embedding(V,E)
        s.gru=nn.GRU(E+Z,H,batch_first=True); s.out=nn.Linear(H,V); s.H=H
    def encode(s,x):
        h=s.emb(x).transpose(1,2); h=F.relu(s.c1(h)); h=F.relu(s.c2(h)); h=F.relu(s.c3(h))
        h=F.relu(s.fc(h.flatten(1))); return s.mu(h),s.lv(h)
    def decode_tf(s,z,x):
        bos=torch.full((x.size(0),1),1,dtype=torch.long); inp=torch.cat([bos,x[:,:-1]],1)
        e=s.demb(inp); zz=z.unsqueeze(1).expand(-1,L,-1)
        h0=torch.tanh(s.zh(z)).unsqueeze(0)
        o,_=s.gru(torch.cat([e,zz],-1),h0); return s.out(o)
    @torch.no_grad()
    def decode(s,z):
        B=z.size(0); h=torch.tanh(s.zh(z)).unsqueeze(0); tok=torch.full((B,1),1,dtype=torch.long); outs=[]
        for t in range(L):
            e=s.demb(tok); o,h=s.gru(torch.cat([e,z.unsqueeze(1)],-1),h); lg=s.out(o[:,0]); tok=lg.argmax(-1,keepdim=True); outs.append(tok)
        return torch.cat(outs,1)
def detok(row):
    r=[]
    for i in row.tolist():
        if i==2: break
        if i>2: r.append(vocab[i])
    return "".join(r)
if __name__=="__main__":
    tr=mk(M["ae_train"]); va=mk(M["ae_val"][:1000]); print("train",tr.shape)
    m=AE(); opt=torch.optim.Adam(m.parameters(),lr=2e-3)
    bs=256; nb=len(tr)//bs; BUDGET=float(__import__("sys").argv[1])*60; t0=time.time(); step=0
    total=int(BUDGET/0.55)  # rough guess of steps; schedule by wall-clock instead
    beta_max=float(__import__("sys").argv[2])
    ep=0; done=False
    while not done:
        perm=torch.randperm(len(tr)); m.train()
        for b in range(nb):
            frac=(time.time()-t0)/BUDGET
            if frac>=1: done=True; break
            lr=2e-3*(0.5*(1+math.cos(math.pi*min(frac,1))))*0.95+1e-4
            for g in opt.param_groups: g["lr"]=lr
            beta=beta_max*min(1,frac/0.3)
            x=tr[perm[b*bs:(b+1)*bs]]
            mu,lv=m.encode(x); z=mu+torch.randn_like(mu)*torch.exp(0.5*lv)
            lg=m.decode_tf(z,x)
            rec=F.cross_entropy(lg.reshape(-1,V),x.reshape(-1),ignore_index=0)
            kl=-0.5*torch.mean(torch.sum(1+lv-mu**2-lv.exp(),1))
            loss=rec*1.0+beta*kl/ L* L * 1.0 if False else rec+beta*kl
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); step+=1
            if step%100==0:
                m.eval()
                with torch.no_grad():
                    mu,_=m.encode(va); lg=m.decode_tf(mu,va); acc=((lg.argmax(-1)==va)|(va==0)).all(1).float().mean().item()
                    tokacc=((lg.argmax(-1)==va)[va!=0]).float().mean().item()
                print(f"ep{ep} step{step} t={time.time()-t0:.0f}s rec={rec.item():.3f} kl={kl.item():.1f} beta={beta:.4f} val_tf_tokacc={tokacc:.4f} val_tf_exact={acc:.3f}",flush=True); m.train()
        ep+=1
        torch.save(m.state_dict(),O+"ae.pt")
    torch.save(m.state_dict(),O+"ae.pt"); print("saved")

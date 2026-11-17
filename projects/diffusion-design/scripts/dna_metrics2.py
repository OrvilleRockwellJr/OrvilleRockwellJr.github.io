import json,sys,numpy as np,torch,pandas as pd
sys.path.insert(0,"/home/claude/build/scripts"); import gd
from sklearn.metrics import roc_auc_score
torch.set_num_threads(1); O="/home/claude/build/models/"; D="/home/claude/build/data/"; NT="ACGT"
P=json.load(open(D+"dna_prep2.json")); C=pd.DataFrame(P["crispron"]); kte=C[(C.lab=="kim")&(C.split=="test")]
heads=[]
for sd in torch.load(O+"dna_critics.pt"): c=gd.Critic(120,sd["l1.weight"].shape[0]); c.load_state_dict(sd); c.eval(); heads.append(c)
def x0(ss):
    a=np.zeros((len(ss),30,4),np.float32)
    for k,s in enumerate(ss):
        for i,ch in enumerate(s): a[k,i,NT.index(ch)]=1
    return torch.tensor(a.reshape(len(ss),-1)*2-1)
X=x0(kte.s.tolist()); y=kte.y.values; out={}
for ti in (0,50,100,150):
    tt=torch.full((len(X),),ti,dtype=torch.long); torch.manual_seed(3); xt,_=gd.q_sample(X,tt)
    with torch.no_grad(): out[ti]=float(roc_auc_score(y,gd.ens_logits(heads,xt,tt).mean(0).numpy()))
E=json.load(open(O+"dna_eval2.json")); J=json.load(open(O+"dna_judge.json"))
m={"n_train":int((C[(C.lab=="kim")].split=="train").sum()),"n_test":int(len(kte)),"judge_n_train":int((C[(C.lab=="luo")].split=="train").sum()),
   "critic_auc_t0":out[0],"critic_auc_t50":out[50],"critic_auc_t100":out[100],"critic_auc_t150":out[150],"judge_auc_test":E["judge@luo_test"]["auc"],"judge_C":J["C"],
   "eval":{k:v for k,v in E.items() if "@" in k},"critic_judge_spearman_prior":E["critic_judge_spearman_prior"],
   "critic_trained_on":E["critic_trained_on"],"judge_trained_on":E["judge_trained_on"]}
json.dump(m,open(O+"dna_train_metrics.json","w"),indent=1); print(json.dumps(m)[:900])

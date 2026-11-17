import json,sys,time,torch
sys.path.insert(0,"/home/claude/build/scripts"); import gd
from train_ae import AE, mk
torch.set_num_threads(2); D="/home/claude/build/data/"; O="/home/claude/build/models/"
M=json.load(open(D+"mol2.json")); ae=AE(); ae.load_state_dict(torch.load(O+"ae.pt")); ae.eval(); st=torch.load(O+"mol_stats.pt")
out=[]
with torch.no_grad():
    for i in range(0,len(M["ae_train"]),1024): out.append(ae.encode(mk(M["ae_train"][i:i+1024]))[0])
X=(torch.cat(out)-st["mean"])/st["std"]; print(X.shape,flush=True)
steps=int(sys.argv[1]); t=time.time()
den=gd.train_denoiser(X,steps=steps,bs=256,lr=1e-3,H=512,log=lambda m:print(m,f"{time.time()-t:.0f}s",flush=True))
torch.save(den.state_dict(),O+"mol_denoiser_long.pt"); print("done",time.time()-t)

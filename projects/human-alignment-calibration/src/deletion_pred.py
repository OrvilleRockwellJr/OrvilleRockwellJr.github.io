"""Post-hoc robustness check (see README, Deviations): deletion test on the PREDICTED class.
The pre-specified anchoring index g uses the true-class probability, which is bounded above by
confidence for wrong predictions (a floor effect). Here the drop is measured on the probability of the
class the model actually chose, so correct and wrong predictions share the same ceiling (= confidence)."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from extract import *

img, hm, lab, idx = build_subset(1500, 0, '/home/claude/data/subset_1500_0.npz')
hmaps = human_maps(hm)
perm = np.random.RandomState(2).permutation(len(lab))
perm = np.where(perm == np.arange(len(lab)), (perm + 1) % len(lab), perm)
MASK = np.stack([(h >= np.quantile(h, 0.8)) for h in hmaps])
os.makedirs('results/raw_pred', exist_ok=True)
for name in MODELS:
    fn = f'results/raw_pred/{name}.npz'
    if os.path.exists(fn): continue
    model = MODELS[name][0](weights=MODELS[name][1]).eval()
    N = len(lab); q0 = np.zeros(N); qh = np.zeros(N); qs = np.zeros(N)
    with torch.no_grad():
        for s in range(0, N, 32):
            e = min(N, s + 32); x = prep(img[s:e])
            p = model(x).softmax(1); pc = p.argmax(1).view(-1, 1)
            q0[s:e] = p.gather(1, pc).squeeze(1).numpy()
            for arr, ids in ((qh, np.arange(s, e)), (qs, perm[s:e])):
                m = torch.from_numpy(MASK[ids]).unsqueeze(1).float()
                arr[s:e] = model(x * (1 - m)).softmax(1).gather(1, pc).squeeze(1).numpy()
    np.savez(fn, q0=q0, q_h=qh, q_s=qs)
    print(name, 'done', flush=True)

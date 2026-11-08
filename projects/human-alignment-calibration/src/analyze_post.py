"""Post-hoc check H3b: anchoring measured on the predicted class (see README, Deviations)."""
import json, glob, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from analyze import perm_test, cv_delta_auc, holm
S = json.load(open('results/summary.json')); out = {}
for n in S['models']:
    d = np.load(f'results/raw/{n}.npz'); q = np.load(f'results/raw_pred/{n}.npz')
    y = d['correct'].astype(int); c = d['conf']; g2 = q['q_s'] - q['q_h']
    t, p = perm_test(g2, y, c); dA, ci, _ = cv_delta_auc(g2, y, c)
    out[n] = {'mean_g_pred': float(g2.mean()), 'within_bin_auroc': float(t), 'p': float(p),
              'delta_cv_auroc': float(dA), 'delta_ci95': [float(ci[0]), float(ci[1])]}
adj = holm(np.array([out[n]['p'] for n in out]))
for n, a in zip(out, adj): out[n]['p_holm'] = float(a)
json.dump(out, open('results/summary_posthoc_H3b.json', 'w'), indent=1)
for n, m in out.items(): print(n, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})

"""Pre-specified analyses (see PREREGISTRATION.md). Reads results/raw/*.npz, writes results/summary.json + figures."""
import glob, json, os, sys
import numpy as np
from scipy.stats import spearmanr, rankdata, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

CEIL = 0.65753; B = 5000; NB = 10
rng = np.random.RandomState(0)

def ece(conf, y, nb=15):
    bins = np.minimum((conf * nb).astype(int), nb - 1); e = 0
    for b in range(nb):
        m = bins == b
        if m.any(): e += m.mean() * abs(y[m].mean() - conf[m].mean())
    return e

def make_bins(c):
    q = np.quantile(c, np.linspace(0, 1, NB + 1)[1:-1])
    return np.searchsorted(q, c, side='right')

def auc_from_ranks(r, y):
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0), n1 * n0

def stat(a, y, bins):
    num = den = 0
    for b in range(NB):
        m = bins == b; yy = y[m]
        if yy.sum() == 0 or yy.sum() == len(yy): continue
        au, w = auc_from_ranks(rankdata(a[m]), yy); num += au * w; den += w
    return num / den

def perm_test(a, y, c):
    bins = make_bins(c); t = stat(a, y, bins)
    idx = [np.where(bins == b)[0] for b in range(NB)]
    null = np.empty(B)
    for i in range(B):
        ap = a.copy()
        for ix in idx: ap[ix] = a[rng.permutation(ix)]
        null[i] = stat(ap, y, bins)
    p = (1 + np.sum(np.abs(null - 0.5) >= abs(t - 0.5))) / (B + 1)
    return t, p

def cv_delta_auc(a, y, c, reps=5, nboot=1000):
    lc = np.log(np.clip(c, 1e-6, 1 - 1e-6) / (1 - np.clip(c, 1e-6, 1 - 1e-6)))
    X0 = lc[:, None]; X1 = np.c_[lc, (a - a.mean()) / a.std()]
    o0 = np.zeros(len(y)); o1 = np.zeros(len(y))
    for r in range(reps):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X0, y):
            o0[te] += LogisticRegression(max_iter=1000).fit(X0[tr], y[tr]).decision_function(X0[te]) / reps
            o1[te] += LogisticRegression(max_iter=1000).fit(X1[tr], y[tr]).decision_function(X1[te]) / reps
    d = roc_auc_score(y, o1) - roc_auc_score(y, o0)
    bs = []
    for _ in range(nboot):
        i = rng.randint(0, len(y), len(y))
        if y[i].min() == y[i].max(): continue
        bs.append(roc_auc_score(y[i], o1[i]) - roc_auc_score(y[i], o0[i]))
    return d, np.percentile(bs, [2.5, 97.5]), roc_auc_score(y, o0)

def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, (m - k) * ps[i]); adj[i] = min(1, run)
    return adj

def main():
    files = sorted(glob.glob('results/raw/*.npz'))
    R = {os.path.basename(f)[:-4]: dict(np.load(f)) for f in files}
    names = list(R)
    S = {'n_images': int(len(next(iter(R.values()))['label'])), 'models': {}}
    for n, d in R.items():
        y = d['correct'].astype(int); c = d['conf']; e = d['rho'] - d['null']
        g = d['p_del_s'] - d['p_del_h']          # anchoring index
        m = {'top1': float(y.mean()), 'ece': float(ece(c, y)), 'mean_conf': float(c.mean()),
             'rho': float(d['rho'].mean()), 'null': float(d['null'].mean()),
             'align_norm': float(d['rho'].mean() / CEIL), 'excess_norm': float(e.mean() / CEIL),
             'rho_gt_null_p': float(wilcoxon(d['rho'], d['null']).pvalue),
             'mean_anchoring': float(g.mean()), 'n_err': int((1 - y).sum())}
        print(n, m, flush=True)
        for tag, a in (('H2_excess_alignment', e), ('H3_anchoring', g)):
            t, p = perm_test(a, y, c); dA, ci, a0 = cv_delta_auc(a, y, c)
            m[tag] = {'within_bin_auroc': float(t), 'p': float(p), 'delta_cv_auroc': float(dA),
                      'delta_ci95': [float(ci[0]), float(ci[1])], 'auroc_conf_only': float(a0)}
        # ECE by tercile of excess alignment, bootstrap CI
        ter = np.digitize(e, np.quantile(e, [1 / 3, 2 / 3])); tr = {}
        for k in range(3):
            mk = ter == k; bs = []
            for _ in range(500):
                i = rng.choice(np.where(mk)[0], mk.sum())
                bs.append(ece(c[i], y[i]))
            tr[k] = {'ece': float(ece(c[mk], y[mk])), 'ci95': np.percentile(bs, [2.5, 97.5]).tolist(),
                     'acc': float(y[mk].mean()), 'conf': float(c[mk].mean())}
        m['ece_by_tercile'] = tr
        S['models'][n] = m
    for tag in ('H2_excess_alignment', 'H3_anchoring'):
        adj = holm(np.array([S['models'][n][tag]['p'] for n in names]))
        for n, a in zip(names, adj): S['models'][n][tag]['p_holm'] = float(a)
    acc = np.array([S['models'][n]['top1'] for n in names]); al = np.array([S['models'][n]['align_norm'] for n in names])
    ec = np.array([S['models'][n]['ece'] for n in names])
    def resid(v, x): b = np.polyfit(x, v, 1); return v - np.polyval(b, x)
    S['V1_acc_vs_alignment_spearman'] = float(spearmanr(acc, al)[0])
    S['H1_alignment_vs_ece_spearman'] = float(spearmanr(al, ec)[0])
    S['H1_alignment_vs_ece_partial_on_acc'] = float(spearmanr(resid(al, acc), resid(ec, acc))[0])
    json.dump(S, open('results/summary.json', 'w'), indent=1)

    os.makedirs('results/figures', exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for n in names: ax[0].scatter(acc[names.index(n)], al[names.index(n)]); ax[0].annotate(n, (acc[names.index(n)], al[names.index(n)]), fontsize=7)
    ax[0].set_xlabel('top-1 accuracy (ClickMe subset)'); ax[0].set_ylabel('normalised alignment (rho / 0.658)'); ax[0].set_title('Accuracy vs. human alignment')
    for i, n in enumerate(names):
        h = S['models'][n]['H2_excess_alignment']; ax[1].errorbar(h['delta_cv_auroc'], i, xerr=[[h['delta_cv_auroc'] - h['delta_ci95'][0]], [h['delta_ci95'][1] - h['delta_cv_auroc']]], fmt='o', color='C0')
        h = S['models'][n]['H3_anchoring']; ax[1].errorbar(h['delta_cv_auroc'], i + 0.25, xerr=[[h['delta_cv_auroc'] - h['delta_ci95'][0]], [h['delta_ci95'][1] - h['delta_cv_auroc']]], fmt='s', color='C3')
    ax[1].axvline(0, color='k', lw=.8); ax[1].set_yticks(range(len(names))); ax[1].set_yticklabels(names, fontsize=8)
    ax[1].set_xlabel('gain in error-detection AUROC over confidence alone'); ax[1].set_title('blue: excess alignment (H2), red: anchoring (H3)')
    for i, n in enumerate(names):
        t = S['models'][n]['ece_by_tercile']; ax[2].plot([0, 1, 2], [t[k]['ece'] for k in range(3)], marker='o', label=n)
    ax[2].set_xticks([0, 1, 2]); ax[2].set_xticklabels(['low', 'mid', 'high']); ax[2].set_xlabel('excess-alignment tercile (within model)'); ax[2].set_ylabel('ECE'); ax[2].legend(fontsize=6)
    plt.tight_layout(); plt.savefig('results/figures/main.png', dpi=160)
    print('done')


if __name__ == '__main__':
    main()

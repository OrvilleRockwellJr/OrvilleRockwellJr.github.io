import json
S = json.load(open('results/summary.json')); P = json.load(open('results/summary_posthoc_H3b.json'))
L = ['| Model | Top-1 | ECE | Mean conf. | Alignment (norm.) | Excess alignment (norm.) | H2 within-bin AUROC (Holm p) | H2 ΔAUROC [95% CI] | H3 AUROC (Holm p) | H3b AUROC (Holm p) |', '|---|---|---|---|---|---|---|---|---|---|']
for n, m in S['models'].items():
    a, b, c = m['H2_excess_alignment'], m['H3_anchoring'], P[n]
    L.append(f"| {n} | {m['top1']:.3f} | {m['ece']:.3f} | {m['mean_conf']:.3f} | {m['align_norm']:.3f} | {m['excess_norm']:.3f} | {a['within_bin_auroc']:.3f} ({a['p_holm']:.2f}) | {a['delta_cv_auroc']:+.4f} [{a['delta_ci95'][0]:+.4f}, {a['delta_ci95'][1]:+.4f}] | {b['within_bin_auroc']:.3f} ({b['p_holm']:.3f}) | {c['within_bin_auroc']:.3f} ({c['p_holm']:.3f}) |")
L.append(''); L.append(f"Accuracy vs. normalised alignment, Spearman across models: {S['V1_acc_vs_alignment_spearman']:+.2f} (n = 8)")
L.append(f"Alignment vs. ECE, Spearman: {S['H1_alignment_vs_ece_spearman']:+.2f}; after residualising on accuracy: {S['H1_alignment_vs_ece_partial_on_acc']:+.2f}")
open('results/RESULTS.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

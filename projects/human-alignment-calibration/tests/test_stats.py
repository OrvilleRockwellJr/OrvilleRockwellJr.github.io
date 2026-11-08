import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import analyze as A
from extract import pearson, rank_flat

def test_auc_matches_sklearn():
    r = np.random.RandomState(0); y = r.randint(0, 2, 300); a = r.randn(300) + y * .5
    au, _ = A.auc_from_ranks(rankdata(a), y)
    assert abs(au - roc_auc_score(y, a)) < 1e-12

def test_spearman_with_ties_uses_average_ranks():
    a = np.array([1, 1, 1, 2.0]); b = np.array([1, 2, 3, 4.0])
    from scipy.stats import spearmanr
    assert abs(pearson(rank_flat(a), rank_flat(b)) - spearmanr(a, b)[0]) < 1e-12

def test_ece_zero_when_calibrated():
    r = np.random.RandomState(1); c = r.uniform(.05, .95, 200000); y = (r.rand(len(c)) < c).astype(int)
    assert A.ece(c, y) < 0.01

def test_perm_test_null_and_signal():
    A.B = 500
    r = np.random.RandomState(2); c = r.uniform(.1, .9, 1500); y = (r.rand(1500) < c).astype(int)
    _, p0 = A.perm_test(r.randn(1500), y, c)            # alignment unrelated to y given c
    _, p1 = A.perm_test(r.randn(1500) + 1.0 * y, y, c)  # alignment carries signal
    assert p0 > 0.01 and p1 < 0.01

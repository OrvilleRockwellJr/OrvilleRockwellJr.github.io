"""Per-image extraction: predictions, confidence, saliency alignment, null alignment, deletion test."""
import sys, os, time, glob, json, argparse
import numpy as np, torch, torch.nn.functional as F, torchvision.models as tvm
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(__file__))
from tfrec import read_records, parse_example
from PIL import Image
import io

torch.set_num_threads(2)
DATA = os.environ.get('CLICKME_DIR', '/home/claude/data/clickme')
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255

MODELS = {
    'alexnet': (tvm.alexnet, tvm.AlexNet_Weights.IMAGENET1K_V1),
    'vgg16': (tvm.vgg16, tvm.VGG16_Weights.IMAGENET1K_V1),
    'resnet18': (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1),
    'resnet50': (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V1),
    'densenet121': (tvm.densenet121, tvm.DenseNet121_Weights.IMAGENET1K_V1),
    'efficientnet_b0': (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights.IMAGENET1K_V1),
    'convnext_tiny': (tvm.convnext_tiny, tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    'vit_b_16': (tvm.vit_b_16, tvm.ViT_B_16_Weights.IMAGENET1K_V1),
}

def build_subset(n, seed, cache):
    if os.path.exists(cache):
        z = np.load(cache); return z['img'], z['hm'], z['lab'], z['idx']
    shards = sorted(glob.glob(f'{DATA}/val-*.tfrecords'), key=lambda p: int(p.split('-')[-1].split('.')[0]))
    recs = [(s, k) for s in shards for k, _ in enumerate(read_records(s))]
    rng = np.random.RandomState(seed)
    pick = set(rng.choice(len(recs), n, replace=False).tolist())
    imgs, hms, labs, idx = [], [], [], []
    g = 0
    for s in shards:
        for d in read_records(s):
            if g in pick:
                e = parse_example(d)
                imgs.append(np.array(Image.open(io.BytesIO(e['image'])).convert('RGB')))
                hms.append(np.array(Image.open(io.BytesIO(e['heatmap'])).convert('L')))
                labs.append(int(e['label'])); idx.append(g)
            g += 1
    img, hm, lab, idx = np.stack(imgs), np.stack(hms), np.array(labs), np.array(idx)
    np.savez_compressed(cache, img=img, hm=hm, lab=lab, idx=idx)
    return img, hm, lab, idx

def gkernel(size=10, sigma=10):
    r = torch.arange(-(size - 1) // 2, (size - 1) // 2 + 1, dtype=torch.float32)  # same as released code
    xs, ys = torch.meshgrid(r, r.flip(0), indexing='xy')
    k = torch.exp(-(xs ** 2 + ys ** 2) / (2 * sigma ** 2)); k = k / k.sum()
    return k.view(1, 1, size, size)

def human_maps(hm):  # (N,256,256) uint8 -> (N,224,224) float, lab preprocessing
    h = torch.from_numpy(hm).float().unsqueeze(1)
    h = F.interpolate(h, size=(64, 64), mode='bilinear', align_corners=False)
    k = gkernel()
    h = F.pad(h, (4, 5, 4, 5)); h = F.conv2d(h, k)       # 'SAME' for even kernel
    h = F.interpolate(h, size=(224, 224), mode='bilinear', align_corners=False)
    return h.squeeze(1).numpy()

def prep(img):  # (B,256,256,3) uint8 -> normalised (B,3,224,224)
    x = torch.from_numpy(img).permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
    return (x - MEAN) / STD

def rank_flat(m):
    return rankdata(m.reshape(-1), method='average').astype(np.float64)

def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30))

def run(name, img, hm_raw, lab, hmaps, perm, null_k=3, bs=16, frac=0.2):
    ctor, w = MODELS[name]
    model = ctor(weights=w).eval()
    for p in model.parameters(): p.requires_grad_(False)
    N = len(lab)
    out = {k: np.zeros(N) for k in ['conf', 'correct', 'ptrue', 'rho', 'null', 'p_del_h', 'p_del_s', 'rho_pred']}
    hrank = [rank_flat(h) for h in hmaps]
    rng = np.random.RandomState(1)
    nulls = np.stack([rng.permutation(N) for _ in range(null_k)], 1)
    t0 = time.time()
    for s in range(0, N, bs):
        e = min(N, s + bs); b = e - s
        x = prep(img[s:e]).requires_grad_(True)
        y = torch.from_numpy(lab[s:e])
        logits = model(x)
        prob = logits.softmax(1).detach()
        sel = logits.gather(1, y.view(-1, 1)).sum()
        g, = torch.autograd.grad(sel, x)
        sal = g.abs().amax(1).numpy()
        out['conf'][s:e] = prob.max(1).values.numpy()
        out['correct'][s:e] = (prob.argmax(1) == y).numpy()
        out['ptrue'][s:e] = prob.gather(1, y.view(-1, 1)).squeeze(1).numpy()
        for j in range(b):
            i = s + j
            r = rank_flat(sal[j])
            out['rho'][i] = pearson(r, hrank[i])
            out['null'][i] = np.mean([pearson(r, hrank[k]) for k in nulls[i]])
        # deletion test (forward only): mask top-20% of own human map vs. another image's map
        with torch.no_grad():
            for key, ids in (('p_del_h', np.arange(s, e)), ('p_del_s', perm[s:e])):
                m = np.stack([(hmaps[k] >= np.quantile(hmaps[k], 1 - frac)) for k in ids])
                mt = torch.from_numpy(m).unsqueeze(1).float()
                xm = x.detach() * (1 - mt)  # 0 == dataset mean colour after normalisation
                out[key][s:e] = model(xm).softmax(1).gather(1, y.view(-1, 1)).squeeze(1).numpy()
        if (s // bs) % 10 == 0:
            print(f'{name} {e}/{N} {time.time()-t0:.0f}s', flush=True)
    out['top1'] = out['correct'].mean()
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1500); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--models', default=','.join(MODELS)); ap.add_argument('--out', default='results/raw')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    img, hm, lab, idx = build_subset(a.n, a.seed, f'/home/claude/data/subset_{a.n}_{a.seed}.npz')
    hmaps = human_maps(hm)
    perm = np.random.RandomState(2).permutation(len(lab))
    perm = np.where(perm == np.arange(len(lab)), (perm + 1) % len(lab), perm)
    for name in a.models.split(','):
        fn = f'{a.out}/{name}.npz'
        if os.path.exists(fn): continue
        o = run(name, img, hm, lab, hmaps, perm)
        np.savez(fn, idx=idx, label=lab, **{k: np.asarray(v) for k, v in o.items()})
        print(name, 'top1', o['top1'], 'mean rho', o['rho'].mean(), 'null', o['null'].mean(), flush=True)

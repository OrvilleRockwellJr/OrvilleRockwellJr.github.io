"""Minimal TFRecord + tf.Example reader (no TensorFlow dependency)."""
import struct, io
import numpy as np
from PIL import Image

def _varint(b, i):
    r = s = 0
    while True:
        c = b[i]; i += 1
        r |= (c & 0x7F) << s
        if not c & 0x80: return r, i
        s += 7

def _fields(b):
    i = 0
    while i < len(b):
        k, i = _varint(b, i); f, w = k >> 3, k & 7
        if w == 0: v, i = _varint(b, i)
        elif w == 2:
            n, i = _varint(b, i); v = b[i:i+n]; i += n
        elif w == 1: v = b[i:i+8]; i += 8
        elif w == 5: v = b[i:i+4]; i += 4
        else: raise ValueError(w)
        yield f, v

def parse_example(buf):
    out = {}
    for f, feats in _fields(buf):            # Example.features
        for f2, entry in _fields(feats):     # Features.feature map entries
            key = val = None
            for f3, x in _fields(entry):
                if f3 == 1: key = x.decode()
                else: val = x
            for f4, lst in _fields(val):     # Feature oneof
                inner = list(_fields(lst))
                if f4 == 1: out[key] = inner[0][1]                     # bytes_list
                elif f4 == 3:                                          # int64_list (packed)
                    p = inner[0][1]; i = 0
                    if isinstance(p, int): out[key] = p
                    else: out[key], _ = _varint(p, 0)
    return out

def read_records(path):
    with open(path, 'rb') as fh:
        while True:
            h = fh.read(8)
            if len(h) < 8: return
            n = struct.unpack('<Q', h)[0]; fh.read(4)
            d = fh.read(n); fh.read(4)
            yield d

def load_shard(path):
    for d in read_records(path):
        e = parse_example(d)
        img = np.array(Image.open(io.BytesIO(e['image'])).convert('RGB'))
        hm = np.array(Image.open(io.BytesIO(e['heatmap'])).convert('L'))
        yield img, hm, int(e['label'])

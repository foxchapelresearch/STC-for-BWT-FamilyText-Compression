"""What abstractions did the online LSTM learn? Cosine geometry + average-linkage clustering of its byte embeddings.

input embedding  = how a byte acts as context  (Goodman's "conditional" role)
output embedding = how a byte is predicted     (Goodman's "predictive" role)

usage: python3 -I inspect_emb.py <data file> <maxbytes> <prefix>
"""
import sys, numpy as np

data = np.frombuffer(open(sys.argv[1], 'rb').read()[:int(sys.argv[2])], dtype=np.uint8)
cnt = np.bincount(data, minlength=256)
keep = [b for b in range(256) if cnt[b] >= 2000]
lab = lambda b: {32: 'SP', 10: 'NL', 9: 'TAB'}.get(b, chr(b) if 33 <= b < 127 else f'x{b:02X}')

def unit(E):
    E = E - E[keep].mean(0)
    return E / np.linalg.norm(E, axis=1, keepdims=True)

def mean_cos(U, A, Bs=None):
    if Bs is None:
        s = [U[a] @ U[b] for i, a in enumerate(A) for b in A[i + 1:]]
    else:
        s = [U[a] @ U[b] for a in A for b in Bs]
    return float(np.mean(s))

def cluster(U, k):
    groups = [[b] for b in keep]
    while len(groups) > k:
        best = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                d = np.mean([1 - U[a] @ U[b] for a in groups[i] for b in groups[j]])
                if best is None or d < best[0]: best = (d, i, j)
        _, i, j = best
        groups[i] += groups.pop(j)
    return groups

digits = [b for b in range(48, 58) if b in keep]
lower = [b for b in range(97, 123) if b in keep]
upper = [b for b in range(65, 91) if b in keep and b + 32 in keep]
others = [b for b in keep if b not in digits]
for name in ('in', 'out'):
    U = unit(np.load(f'{sys.argv[3]}_emb_{name}.npy'))
    print(f"=== {name}put embedding ({'context role' if name == 'in' else 'predicted role'})")
    print(f"  mean cos digit-digit {mean_cos(U, digits):+.3f}   digit-other {mean_cos(U, digits, others):+.3f}"
          f"   letter-letter {mean_cos(U, lower):+.3f}")
    pair = np.mean([U[b] @ U[b + 32] for b in upper])
    rnd = np.mean([U[b] @ U[c + 32] for b in upper for c in upper if c != b])
    print(f"  case pairs (A,a) {pair:+.3f}  vs  mismatched (A,b) {rnd:+.3f}")
    for q in [ord('7'), ord('0'), ord('1'), ord('A'), ord('e'), ord('['), 0xC3, 0xA9]:
        if q not in keep: continue
        nb = sorted((b for b in keep if b != q), key=lambda b: -(U[q] @ U[b]))[:6]
        print(f"  nearest to {lab(q):>4}: " + ' '.join(lab(b) for b in nb))
    print("  14 clusters (average linkage, cosine):")
    for g in sorted(cluster(U, 14), key=lambda g: -sum(cnt[b] for b in g)):
        print("    {" + ' '.join(lab(b) for b in sorted(g)) + "}")

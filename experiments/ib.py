# Agglomerative Information Bottleneck on the byte alphabet (Slonim & Tishby 1999):
# merging symbols a,b loses (p_a+p_b) * JS_pi(P(Y|a), P(Y|b)) bits of I(T;Y), Y = (prev byte, next byte).
import sys, numpy as np
x = np.frombuffer(open(sys.argv[1], 'rb').read(), dtype=np.uint8)
nxt = np.zeros((256, 256)); prv = np.zeros((256, 256))
np.add.at(nxt, (x[:-1], x[1:]), 1); np.add.at(prv, (x[1:], x[:-1]), 1)
cnt = nxt.sum(1)
keep = [a for a in range(256) if cnt[a] >= 2000]
lab = lambda a: chr(a) if 33 <= a < 127 else {32: 'SP', 10: 'NL'}.get(a, f'x{a:02x}')
clusters = [[a] for a in keep]
Pn = [nxt[a].copy() for a in keep]; Pp = [prv[a].copy() for a in keep]
N = x.size
def H(v):
    v = v[v > 0]; s = v.sum(); q = v / s
    return -(q * np.log2(q)).sum()
def js_cost(i, j):
    c = 0.0
    for P in (Pn, Pp):
        a, b = P[i], P[j]; na, nb = a.sum(), b.sum()
        c += ((na + nb) * H(a + b) - na * H(a) - nb * H(b)) / N   # = (p_a+p_b) JS_pi
    return c
import itertools
cost = {}
for i, j in itertools.combinations(range(len(clusters)), 2): cost[(i, j)] = js_cost(i, j)
alive = set(range(len(clusters))); step = 0; total = 0
print(f"{len(keep)} symbols with count>=2000; merges in order of information lost (bits/char):")
while len(alive) > 12:
    (i, j), c = min(((k, v) for k, v in cost.items()), key=lambda kv: kv[1])
    total += c; step += 1
    a_lab = ''.join(lab(s) for s in clusters[i]); b_lab = ''.join(lab(s) for s in clusters[j])
    if step <= 45 or len(alive) <= 20:
        print(f"  {step:3d}. loss {c*1e3:7.3f} mbit/char  [{a_lab}] + [{b_lab}]")
    clusters[i] = clusters[i] + clusters[j]; Pn[i] = Pn[i] + Pn[j]; Pp[i] = Pp[i] + Pp[j]
    alive.discard(j)
    cost = {k: v for k, v in cost.items() if j not in k and i not in k}
    for k in alive:
        if k != i: cost[(min(i, k), max(i, k))] = js_cost(min(i, k), max(i, k))
print("final clusters:")
for k in sorted(alive, key=lambda k: -sum(cnt[s] for s in clusters[k])):
    print("  ", ''.join(lab(s) for s in sorted(clusters[k])))

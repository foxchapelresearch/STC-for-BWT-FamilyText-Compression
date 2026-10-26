"""Compare string-context digit coding (dcm per-run costs) with explicit value models on XML header fields.

Value models (causal, adaptive, ideal code length):
  page id   : delta from previous page id; Elias-gamma bucket coded adaptively, mantissa bits raw.
  timestamp : predicted by interpolating already-decoded (revision id -> timestamp) pairs; residual coded
              with an adaptive sign+bucket model, mantissa bits raw.
"""
import sys, struct, re, math, bisect, datetime, collections

x = open(sys.argv[1], 'rb').read()
percost = dict(struct.iter_unpack('<qf', open(sys.argv[2], 'rb').read()))

def line_of(a):
    s = x.rfind(b'\n', 0, a) + 1; e = x.find(b'\n', a)
    return s, x[s:e]

def field(a):
    s, ln = line_of(a); t = ln.strip()
    if t.startswith(b'<timestamp>'): return 'timestamp'
    if t.startswith(b'<id>'):
        ps, prev = line_of(s - 1)
        if b'</title>' in prev: return 'page id'
        if b'<revision>' in prev: return 'revision id'
        return 'contributor id'
    if t.startswith(b'<ip>'): return 'contributor ip'
    return 'text'

cost = collections.Counter(); runs = collections.Counter()
for a, c in percost.items():
    f = field(a); cost[f] += c; runs[f] += 1
print("string-context digit model (dcm, all contexts) by field:")
for f, c in cost.most_common():
    print(f"  {f:15s} runs {runs[f]:9,d}  {c/8:10,.0f} bytes  {c/runs[f]:6.2f} bits/run")

class Adaptive:
    def __init__(self, k): self.c = [0.5] * k; self.t = 0.5 * k
    def cost(self, s):
        b = -math.log2(self.c[s] / self.t); self.c[s] += 1; self.t += 1; return b

def gamma_cost(v, model):  # v >= 1
    k = v.bit_length() - 1
    return model.cost(min(k, 63)) + k

pages = [(mo.group(1), mo.group(2), mo.group(3)) for mo in re.finditer(
    rb'<page>\s*<title>[^<]*</title>\s*<id>(\d+)</id>\s*<revision>\s*<id>(\d+)</id>\s*<timestamp>([^<]+)</timestamp>', x)]
pid = [int(p[0]) for p in pages]; rid = [int(p[1]) for p in pages]
ts = [int(datetime.datetime.strptime(p[2].decode(), '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()) for p in pages]

m_pid = Adaptive(64); bits_pid = 0; prev = 0
for p in pid:
    bits_pid += gamma_cost(p - prev if p > prev else 1, m_pid) + (0 if p > prev else 32); prev = p

m_sign, m_ts = Adaptive(3), Adaptive(64); bits_ts = 0
keys, vals = [], []
for r, t in zip(rid, ts):
    i = bisect.bisect_left(keys, r)
    if keys:
        if 0 < i < len(keys):
            r0, t0, r1, t1 = keys[i-1], vals[i-1], keys[i], vals[i]
            pred = t0 + (t1 - t0) * (r - r0) // max(1, r1 - r0)
        else:
            pred = vals[i-1] if i > 0 else vals[0]
    else:
        pred = 1.15e9
    e = t - int(pred)
    bits_ts += m_sign.cost(0 if e == 0 else 1 if e > 0 else 2)
    if e: bits_ts += gamma_cost(abs(e), m_ts)
    keys.insert(i, r); vals.insert(i, t)

print(f"\npages parsed: {len(pages):,}")
print(f"page id   : string model {cost['page id']/8:9,.0f} bytes  ->  value model (delta) {bits_pid/8:9,.0f} bytes")
print(f"timestamp : string model {cost['timestamp']/8:9,.0f} bytes  ->  value model (rev-id interpolation) {bits_ts/8:9,.0f} bytes")
print(f"            ({cost['timestamp']/len(pages):.1f} -> {bits_ts/len(pages):.1f} bits per timestamp)")

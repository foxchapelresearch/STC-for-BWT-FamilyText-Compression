import sys, re, math, collections, datetime
x = open(sys.argv[1], 'rb').read()
pages = re.findall(rb'<page>\s*<title>[^<]*</title>\s*<id>(\d+)</id>\s*<revision>\s*<id>(\d+)</id>\s*<timestamp>([^<]+)</timestamp>', x)
print("pages parsed:", len(pages))
pid = [int(p[0]) for p in pages]; rid = [int(p[1]) for p in pages]
ts = [datetime.datetime.strptime(p[2].decode(), '%Y-%m-%dT%H:%M:%SZ').timestamp() for p in pages]
inc = sum(b > a for a, b in zip(pid, pid[1:]))
print(f"page ids increasing: {inc}/{len(pid)-1}; median delta {sorted(b-a for a,b in zip(pid,pid[1:]))[len(pid)//2]}")
# cost of page id as log2(delta) gamma-ish vs plain digits
def gamma_bits(v): return 2 * math.floor(math.log2(v)) + 1 if v > 0 else 64
print(f"page ids: Elias-gamma(delta) ~ {sum(gamma_bits(b-a) for a,b in zip(pid,pid[1:]))/8:,.0f} bytes total")
# revision id vs timestamp: sort by rid, check monotonicity of ts
pairs = sorted(zip(rid, ts))
mono = sum(b[1] >= a[1] for a, b in zip(pairs, pairs[1:]))
print(f"timestamps monotone in revision id: {mono}/{len(pairs)-1} adjacent pairs (sorted by rev id)")
# how well does a monotone fit (piecewise-linear interpolation from neighbours by rev id) predict ts? leave-one-out
import bisect
errs = []
for i in range(1, len(pairs) - 1):
    (r0, t0), (r, t), (r1, t1) = pairs[i - 1], pairs[i], pairs[i + 1]
    pred = t0 + (t1 - t0) * (r - r0) / max(1, r1 - r0)
    errs.append(abs(t - pred))
errs.sort()
q = lambda f: errs[int(f * len(errs))]
print(f"|timestamp - interpolation from rev-id neighbours| seconds: median {q(.5):.0f}, p90 {q(.9):.0f}, p99 {q(.99):.0f}")
span = max(ts) - min(ts)
print(f"timestamp span: {span/86400:.0f} days -> uniform cost {math.log2(span):.1f} bits; residual cost ~ {sum(math.log2(2*e+2) for e in errs)/len(errs):.1f} bits (log2 of 2|err|+2)")

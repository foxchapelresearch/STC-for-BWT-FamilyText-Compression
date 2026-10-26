import sys, re, collections
data = open(sys.argv[1], 'rb').read()
n = len(data)
runs = [(mo.start(), mo.end()) for mo in re.finditer(rb'[0-9]+', data)]
nd = sum(e - s for s, e in runs)
print(f"bytes={n:,} digit_bytes={nd:,} ({100*nd/n:.2f}%) runs={len(runs):,}")
lens = collections.Counter(min(e - s, 12) for s, e in runs)
print("run-length histogram (capped 12):", sorted(lens.items()))
def mask(b): return bytes(57 if 48 <= c <= 57 else c for c in b)
fields = collections.Counter(); fdig = collections.Counter()
for s, e in runs:
    ctx = mask(data[max(0, s - 12):s])
    k = ctx[-8:]
    fields[k] += 1; fdig[k] += e - s
print("top 8-byte masked left contexts by digit bytes:")
for k, v in fdig.most_common(30):
    print(f"  {v:>9,} digits {fields[k]:>8,} runs   {k!r}")

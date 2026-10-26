"""Independent reimplementation of the STC digit-context decomposition (paper Sec. 3.4, Table 1).

Writes main.bin (digits -> '9'), s00..s12 side streams, and raw_digits.bin (digits 0..9 in document order).
"""
import os, re, sys

PH = 57  # '9'
CAP = 12

def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    x = open(src, 'rb').read()
    m = x.translate(bytes(PH if 48 <= c <= 57 else c for c in range(256)))
    n = len(m)
    runs = [(mo.start(), mo.end()) for mo in re.finditer(rb'9+', m)]

    def N(i):
        return m[i] + 1 if 0 <= i < n else 0  # boundary -1 sorts first (+1 offset as in the paper)

    def L(a, t):
        return tuple(N(a - 1 - k) for k in range(t))

    def key(a, e, b):
        if b == 1: return L(a, 3) + (N(e),)
        if b == 2: return L(a, 4) + (N(e), N(e + 1))
        if b == 3: return (N(a - 1), N(e))
        if b == 5: return L(a, 4)
        if b == 6: return L(a, 4) + (N(e),)
        if b == 7: return L(a, 8)
        if b == 8: return L(a, 6)
        if b == 9: return (N(e), N(a - 1))
        if b == 10: return L(a, 6)
        if b == 12: return L(a, 6)
        return (N(a - 1),)  # bucket 11

    PACK = {1: 'raw', 2: 'pair', 3: 'pair', 4: 'int', 5: 'int', 6: 'pair', 7: 'raw',
            8: 'pair', 9: 'raw', 10: 'pair', 11: 'raw', 12: 'raw'}

    def digits(a, e):
        return [c - 48 for c in x[a:e]]

    def pack(ds, mode):
        if mode == 'raw':
            return bytes(ds)
        if mode == 'pair':
            o, s = [], 0
            if len(ds) % 2:
                o.append(100 + ds[0]); s = 1
            o += [ds[i] * 10 + ds[i + 1] for i in range(s, len(ds), 2)]
            return bytes(o)
        v = int(''.join(map(str, ds)))
        k = (len(ds) + 1) // 2
        return v.to_bytes(k, 'big')

    by_b = {b: [] for b in range(1, CAP + 1)}
    for a, e in runs:
        by_b[min(e - a, CAP)].append((a, e))

    streams = [bytearray() for _ in range(CAP + 1)]
    for b in range(1, CAP + 1):
        rs = by_b[b]
        if b == 4:
            first = sorted(rs, key=lambda r: N(r[0] - 1))  # stable
            for a, e in first:
                streams[3].append(pack(digits(a, e), 'int')[0])
            second = sorted(rs, key=lambda r: pack(digits(*r), 'int')[0])
            for a, e in second:
                streams[4].append(pack(digits(a, e), 'int')[1])
            continue
        idx = b - 1 if b < 4 else b
        for a, e in sorted(rs, key=lambda r: key(r[0], r[1], b)):
            streams[idx] += pack(digits(a, e), PACK[b])

    open(os.path.join(out, 'main.bin'), 'wb').write(m)
    for i, s in enumerate(streams):
        open(os.path.join(out, f's{i:02d}.bin'), 'wb').write(s)
    raw = bytes(c - 48 for c in x if 48 <= c <= 57)
    open(os.path.join(out, 'raw_digits.bin'), 'wb').write(raw)
    print(f"runs={len(runs):,} digits={len(raw):,} side_raw_bytes={sum(map(len, streams)):,}")
    print("stream sizes:", [len(s) for s in streams])

main()

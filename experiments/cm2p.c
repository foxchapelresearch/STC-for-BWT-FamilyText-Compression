// cm2.c — context mixing with indirect (bit-history) modeling, long orders and abstraction levels.
//
// What changed vs. cmlat.c, aimed at the reasons a semi-static suffix-tree coder (M03) wins:
//   * statistic sharing: every context slot keeps a small non-stationary bit history (n0, n1 <= 15);
//     its probability comes from a StateMap shared by all contexts of that type with the same history
//     (indirect modeling), alongside the slot's own direct estimate;
//   * longer contexts: orders 1-6, 8, 12, 16, 24 plus two word contexts and a match model per level;
//   * 2-way associative hash buckets with checksums (fewer destructive collisions).
// Abstraction levels: 0 literal, 1 digits -> '9', 2 digits + case fold. Abstracted contexts identical
// to the next-finer enabled level are skipped (no duplicate evidence).
// Two mixers (by match length / by number of known contexts), two APMs. Ideal code length.
//
// usage: cm2 <file> <levels e.g. 0 | 1 | 01> [maxbytes]
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef TB
#define TB 27
#endif
#define NORD 10
#define NCTX (NORD + 2)
#define MB 22
#define MINLEN 7
static const int ORD[NORD] = {1, 2, 3, 4, 5, 6, 8, 12, 16, 24};

typedef struct { uint16_t p; uint8_t st, chk; } Slot;
static uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL; return z ^ (z >> 31);
}
static float stretchf(float p) { return logf(p / (1 - p)); }
static float squashf(float t) { if (t > 20) t = 20; if (t < -20) t = -20; return 1 / (1 + expf(-t)); }
static int isletter(int b) { return (b | 32) >= 'a' && (b | 32) <= 'z'; }
static uint8_t ab(uint8_t b, int a) {
    if (a >= 1 && b >= '0' && b <= '9') b = '9';
    if (a >= 2 && b >= 'A' && b <= 'Z') b += 32;
    return b;
}
// non-stationary bit history: n0, n1 in 0..15; on a bit, bump its count and discount the other
static uint8_t st_next(uint8_t s, int y) {
    int n0 = s & 15, n1 = s >> 4;
    if (y) { if (n1 < 15) n1++; if (n0 > 2) n0 = (n0 + 1) / 2; }
    else { if (n0 < 15) n0++; if (n1 > 2) n1 = (n1 + 1) / 2; }
    return (uint8_t)(n0 | n1 << 4);
}
static float rate[1024];

typedef struct { float p; int n; } SM;  // StateMap entry

static float apm_get(float *ap, float pr, int *lo, float *fr) {
    float sv = stretchf(pr); if (sv < -7.999f) sv = -7.999f; if (sv > 7.999f) sv = 7.999f;
    float pos = (sv + 8) * 2; *lo = (int)pos; *fr = pos - *lo;
    return ap[*lo] * (1 - *fr) + ap[*lo + 1] * *fr;
}
static void apm_upd(float *ap, int lo, float fr, int y) {
    ap[lo] += 0.02f * (1 - fr) * (y - ap[lo]); ap[lo + 1] += 0.02f * fr * (y - ap[lo + 1]);
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: cm2 file levels [maxbytes]\n"); return 2; }
    FILE *f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (argc > 3 && atol(argv[3]) > 0 && atol(argv[3]) < n) n = atol(argv[3]);
    uint8_t *x = malloc(n); if (fread(x, 1, n, f) != (size_t)n) return 1; fclose(f);
    int lev[3], nlev = 0; for (const char *c = argv[2]; *c; c++) lev[nlev++] = *c - '0';
    for (int i = 0; i < 1024; i++) rate[i] = 1.0f / (i + 1.5f);
    Slot *tab = calloc((size_t)1 << TB, sizeof(Slot));
    uint64_t tmask = ((uint64_t)1 << TB) - 1;
    uint32_t *mtab[3]; long mptr[3] = {0}; int mlen[3] = {0}; uint8_t *ax[3];
    for (int l = 0; l < nlev; l++) {
        mtab[l] = calloc((size_t)1 << MB, 4);
        ax[l] = malloc(n); for (long i = 0; i < n; i++) ax[l][i] = ab(x[i], lev[l]);
    }
    float mstat[3][64]; for (int l = 0; l < 3; l++) for (int k = 0; k < 64; k++) mstat[l][k] = 0.75f;
    int nctx = 1 + nlev * NCTX;
    SM *sm = malloc(sizeof(SM) * nctx * 256);
    for (int i = 0; i < nctx * 256; i++) {
        int n0 = (i & 255) & 15, n1 = (i & 255) >> 4;
        sm[i].p = (n1 + 0.4f) / (n0 + n1 + 0.8f); sm[i].n = 0;
    }
    int nin = 2 * nctx + nlev + 1;
    int NS1 = 4 * 8, NS2 = (nctx + 1) * 8;
    float *w1 = malloc(sizeof(float) * nin * NS1), *w2 = malloc(sizeof(float) * nin * NS2);
    for (int i = 0; i < nin * NS1; i++) w1[i] = 0.1f;
    for (int i = 0; i < nin * NS2; i++) w2[i] = 0.1f;
    float (*apm1)[33] = malloc(sizeof(float) * 33 * 65536), (*apm2)[33] = malloc(sizeof(float) * 33 * 65536);
    for (int c = 0; c < 65536; c++) for (int k = 0; k < 33; k++) apm1[c][k] = apm2[c][k] = squashf((k - 16) * 0.5f);
    double total = 0, cdig = 0, cother = 0;
    uint64_t base[3][NCTX]; int uniq[3][NCTX];
    uint64_t word[3] = {0}, pword[3] = {0};
    for (long i = 0; i < n; i++) {
        for (int l = 0; l < nlev; l++) {
            uint64_t h = 0x9876 + lev[l] * 0x1111; int j = 0, df = l == 0;
            for (int o = 1; o <= 24; o++) {
                h = mix64(h + (i - o >= 0 ? ax[l][i - o] : 256) + 1);
                if (!df && i - o >= 0 && ax[l][i - o] != ax[l - 1][i - o]) df = 1;
                if (j < NORD && ORD[j] == o) { base[l][j] = h + j; uniq[l][j] = df; j++; }
            }
            int dw = l == 0;
            for (long q = i - 1; !dw && q >= 0 && q > i - 40; q--) { if (ax[l][q] != ax[l - 1][q]) dw = 1; if (!isletter(x[q]) && !(x[q] >= '0' && x[q] <= '9')) break; }
            base[l][NORD] = mix64(word[l] * 31 + (i ? ax[l][i - 1] : 0) + lev[l] * 777 + 11);
            base[l][NORD + 1] = mix64(word[l] + pword[l] * 0x2545F4914F6CDD1DULL + lev[l] * 999 + 12);
            uniq[l][NORD] = uniq[l][NORD + 1] = dw || uniq[l][1];
            if (i >= MINLEN) {
                uint64_t hk = 0; for (int q = 1; q <= MINLEN; q++) hk = hk * 263 + ax[l][i - q];
                hk = mix64(hk) & (((uint64_t)1 << MB) - 1);
                if (mlen[l] == 0) {
                    long cand = mtab[l][hk];
                    if (cand > 0) { int len = 0; while (len < 64 && cand - 1 - len >= 0 && ax[l][cand - 1 - len] == ax[l][i - 1 - len]) len++; if (len >= MINLEN) { mptr[l] = cand; mlen[l] = len; } }
                }
                mtab[l][hk] = (uint32_t)i;
            }
        }
        int c = x[i], partial = 1; double cbyte = 0;
        int mb = mlen[0] == 0 ? 0 : mlen[0] < 16 ? 1 : mlen[0] < 32 ? 2 : 3;
        for (int b = 7; b >= 0; b--) {
            int bit = (c >> b) & 1;
            float st[2 * (1 + 3 * NCTX) + 4]; Slot *sl[1 + 3 * NCTX]; int smi[1 + 3 * NCTX]; int k = 0, ns = 0, known = 0;
            for (int t = 0; t < nctx; t++) {
                Slot *s = NULL;
                if (t == 0) s = &tab[mix64(0x55 + partial) & tmask];
                else {
                    int l = (t - 1) / NCTX, j = (t - 1) % NCTX;
                    if (uniq[l][j]) {
                        uint64_t key = mix64(base[l][j] ^ (partial * 0xD6E8FEB86659FD93ULL));
                        uint8_t chk = (uint8_t)(key >> 56) | 1;
                        Slot *a = &tab[key & tmask], *bb = &tab[(key ^ 1) & tmask];
                        if (a->chk == chk) s = a; else if (bb->chk == chk) s = bb;
                        else {  // replace the less-informed way
                            int na = (a->st & 15) + (a->st >> 4), nb = (bb->st & 15) + (bb->st >> 4);
                            s = na <= nb ? a : bb; s->chk = chk; s->st = 0; s->p = 32768;
                        }
                    }
                }
                if (!s) { st[k++] = 0; st[k++] = 0; smi[ns] = -1; sl[ns++] = NULL; continue; }
                int si = t * 256 + s->st;
                st[k++] = stretchf(sm[si].p < 0.0001f ? 0.0001f : sm[si].p > 0.9999f ? 0.9999f : sm[si].p);
                st[k++] = s->st ? stretchf((s->p + 0.5f) / 65536.0f) : 0;
                if (s->st) known++;
                smi[ns] = si; sl[ns++] = s;
            }
            int mexp[3], midx[3];
            for (int l = 0; l < nlev; l++) {
                mexp[l] = -1; midx[l] = 0; st[k] = 0;
                if (mlen[l] > 0 && !(l > 0 && mlen[l - 1] > 0 && mptr[l - 1] == mptr[l])) {
                    int pb = x[mptr[l]], keep = 0xFF;
                    if (lev[l] >= 1 && pb >= '0' && pb <= '9') keep = 0xF0;
                    if (lev[l] >= 2 && isletter(pb)) keep = 0xDF;
                    int hi = partial ^ (1 << (7 - b));
                    if (((keep >> b) & 1) && ((hi ^ (pb >> (b + 1))) & (keep >> (b + 1))) == 0) {
                        int e = (pb >> b) & 1, ln = mlen[l] > 31 ? 31 : mlen[l];
                        midx[l] = ln * 2 + e; mexp[l] = e;
                        float pc = mstat[l][midx[l]];
                        st[k] = e ? stretchf(pc) : -stretchf(pc);
                    }
                }
                k++;
            }
            st[k++] = 0.3f;
            int s1 = (mb * 8 + (7 - b)) * nin, s2 = ((known > nctx ? nctx : known) * 8 + (7 - b)) * nin;
            float d1 = 0, d2 = 0; for (int q = 0; q < nin; q++) { d1 += w1[s1 + q] * st[q]; d2 += w2[s2 + q] * st[q]; }
            float p1 = squashf(d1), p2 = squashf(d2), pm = squashf((d1 + d2) * 0.5f);
            if (pm < 1e-5f) pm = 1e-5f; if (pm > 1 - 1e-5f) pm = 1 - 1e-5f;
            int c1 = ((i ? x[i - 1] : 0) << 8) | partial;
            int c2 = (int)((mix64(((uint64_t)(i > 1 ? x[i - 2] : 0) << 8 | (i ? x[i - 1] : 0)) * 256 + partial)) & 65535);
            int lo1, lo2; float fr1, fr2;
            float a1 = apm_get(apm1[c1], pm, &lo1, &fr1), a2 = apm_get(apm2[c2], pm, &lo2, &fr2);
            float pf = 0.3f * pm + 0.35f * a1 + 0.35f * a2; if (pf < 1e-5f) pf = 1e-5f; if (pf > 1 - 1e-5f) pf = 1 - 1e-5f;
            double cst = -log2(bit ? pf : 1 - pf); total += cst; cbyte += cst;
            float lr = 0.002f + 0.02f / (1.0f + i / 200000.0f);
            float e1 = bit - p1, e2 = bit - p2;
            for (int q = 0; q < nin; q++) { w1[s1 + q] += lr * e1 * st[q]; w2[s2 + q] += lr * e2 * st[q]; }
            apm_upd(apm1[c1], lo1, fr1, bit); apm_upd(apm2[c2], lo2, fr2, bit);
            for (int q = 0; q < ns; q++) {
                Slot *s = sl[q]; if (!s) continue;
                SM *m = &sm[smi[q]];
                m->p += (bit - m->p) * rate[m->n]; if (m->n < 1023) m->n++;
                int cnt = (s->st & 15) + (s->st >> 4);
                s->p += (int)(((bit ? 65535 : 0) - (int)s->p) * rate[cnt < 30 ? cnt : 30]);
                s->st = st_next(s->st, bit);
            }
            for (int l = 0; l < nlev; l++) if (mexp[l] >= 0) mstat[l][midx[l]] += 0.01f * ((bit == mexp[l]) - mstat[l][midx[l]]);
            partial = partial * 2 + bit;
        }
        if (c >= '0' && c <= '9') cdig += cbyte; else cother += cbyte;
        if ((i + 1) % 50000000 == 0) { fprintf(stderr, "levels=%s %ld MB  %.0f bytes so far (%.4f bpc)\n", argv[2], (i + 1) / 1000000, total / 8, total / (i + 1)); fflush(stderr); }
        for (int l = 0; l < nlev; l++) {
            if (mlen[l] > 0) { if (ax[l][mptr[l]] == ax[l][i]) { mlen[l]++; mptr[l]++; } else mlen[l] = 0; }
            uint8_t a = ax[l][i];
            if (isletter(a)) word[l] = mix64(word[l] + a + 1);
            else if (word[l]) { pword[l] = word[l]; word[l] = 0; }
        }
    }
    printf("levels=%-3s bytes=%ld  total=%.0f bytes (%.4f bpc)  digits=%.0f  other=%.0f\n",
           argv[2], n, total / 8, total / n, cdig / 8, cother / 8);
    return 0;
}

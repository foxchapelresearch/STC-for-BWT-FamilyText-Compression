// cmlat.c — a stronger context-mixing coder (ideal code length) whose contexts can be read through
// abstraction levels: 0 literal, 1 digits -> '9', 2 digits -> '9' + case folded.
//
// Per enabled level: orders 1..7, two word contexts, and a match model that matches in the abstracted
// text (so level 1 finds "same template, different numbers"). Abstract match models only vote on the
// bits their abstraction preserves. Bitwise coding, adaptive probability counters, two logistic mixers,
// final APM. Target is always the exact byte.
//
// usage: cmlat <file> <levels e.g. 0 | 1 | 012> [maxbytes]
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef TB
#define TB 24
#endif
#define NORD 7
#define NCTX (NORD + 2)
#define MAXIN (3 * (NCTX + 1) + 2)
#define MB 22
#define MINLEN 7

typedef struct { uint16_t p; uint8_t n, chk; } Slot;
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
static float rate[256];

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: cmlat file levels [maxbytes]\n"); return 2; }
    FILE *f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (argc > 3 && atol(argv[3]) > 0 && atol(argv[3]) < n) n = atol(argv[3]);
    uint8_t *x = malloc(n); if (fread(x, 1, n, f) != (size_t)n) return 1; fclose(f);
    int gate = getenv("GATE") != NULL;
    int lev[3], nlev = 0; for (const char *c = argv[2]; *c; c++) lev[nlev++] = *c - '0';
    for (int i = 0; i < 256; i++) rate[i] = 1.0f / (i + 1.5f);
    const int LIMIT = 127;
    Slot *tab = calloc((size_t)1 << TB, sizeof(Slot) * 8);  // 8 slots per byte-context bucket region
    uint64_t tmask = ((uint64_t)1 << TB) * 8 - 1;
    // match models, one per level
    uint32_t *mtab[3]; long mptr[3] = {0}; int mlen[3] = {0};
    uint8_t *ax[3];
    for (int l = 0; l < nlev; l++) {
        mtab[l] = calloc((size_t)1 << MB, 4);
        ax[l] = malloc(n); for (long i = 0; i < n; i++) ax[l][i] = ab(x[i], lev[l]);
    }
    uint16_t mstat[3][64][2];  // P(predicted bit correct) by (level, min(len,31)*2+expected bit)
    for (int l = 0; l < 3; l++) for (int k = 0; k < 64; k++) mstat[l][k][0] = 49152, mstat[l][k][1] = 0;
    int nin = nlev * NCTX + nlev + 2;
    float *w1 = malloc(sizeof(float) * nin * 4 * 8), *w2 = malloc(sizeof(float) * nin * 256 * 8);
    for (int i = 0; i < nin * 32; i++) w1[i] = 0.15f;
    for (int i = 0; i < nin * 2048; i++) w2[i] = 0.15f;
    static float apm[256 * 256][33];
    for (int c = 0; c < 256 * 256; c++) for (int k = 0; k < 33; k++) apm[c][k] = squashf((k - 16) * 0.5f);
    double total = 0, cdig = 0, cother = 0;
    uint64_t base[3][NCTX]; int uniq[3][NCTX];
    uint64_t word[3] = {0}, pword[3] = {0};
    for (long i = 0; i < n; i++) {
        // byte-level context hashes per level
        for (int l = 0; l < nlev; l++) {
            uint64_t h = 0x9876 + lev[l] * 0x1111;
            int df = l == 0;
            for (int j = 1; j <= NORD; j++) {
                h = mix64(h + (i - j >= 0 ? ax[l][i - j] : 256) + 1); base[l][j - 1] = h;
                if (!df && i - j >= 0 && ax[l][i - j] != ax[l - 1][i - j]) df = 1;
                uniq[l][j - 1] = df;
            }
            { int dw = l == 0; for (long q = i - 1; !dw && q >= 0 && q > i - 40; q--) { if (ax[l][q] != ax[l - 1][q]) dw = 1; if (!isletter(x[q]) && !(x[q] >= '0' && x[q] <= '9')) break; }
              uniq[l][NORD] = uniq[l][NORD + 1] = dw || uniq[l][NORD - 1]; }
            base[l][NORD] = mix64(word[l] * 31 + (i ? ax[l][i - 1] : 0) + lev[l] * 777 + 1);
            base[l][NORD + 1] = mix64(word[l] + pword[l] * 0x2545F4914F6CDD1DULL + lev[l] * 999 + 2);
            // match model maintenance (abstracted text)
            if (mlen[l] > 0 && mptr[l] < i && ax[l][mptr[l] - 1] == ax[l][i - 1]) { /* extended below */ }
            if (i >= MINLEN) {
                uint64_t hk = 0; for (int j = 1; j <= MINLEN; j++) hk = hk * 263 + ax[l][i - j];
                hk = mix64(hk) & (((uint64_t)1 << MB) - 1);
                if (mlen[l] == 0) {
                    long cand = mtab[l][hk];
                    if (cand > 0) { int len = 0; while (len < 32 && cand - 1 - len >= 0 && ax[l][cand - 1 - len] == ax[l][i - 1 - len]) len++; if (len >= MINLEN) { mptr[l] = cand; mlen[l] = len; } }
                }
                mtab[l][hk] = (uint32_t)i;
            }
        }
        int c = x[i], partial = 1; double cbyte = 0;
        int mbucket = mlen[0] == 0 ? 0 : mlen[0] < 16 ? 1 : mlen[0] < 32 ? 2 : 3;
        for (int b = 7; b >= 0; b--) {
            int bit = (c >> b) & 1;
            float st[MAXIN]; Slot *sl[3 * NCTX + 1]; int k = 0, ns = 0;
            // order-0
            {
                Slot *s = &tab[mix64(0x55 + partial) & tmask];
                sl[ns++] = s; st[k++] = stretchf((s->p + 0.5f) / 65536.0f);
            }
            for (int l = 0; l < nlev; l++) for (int j = 0; j < NCTX; j++) {
                if ((!gate && !uniq[l][j]) || (gate && lev[l] == 0 && !(b <= 3 && (c >> 4) == 3))) { st[k++] = 0; continue; }
                uint64_t key = mix64(base[l][j] ^ (partial * 0xD6E8FEB86659FD93ULL));
                Slot *s = &tab[key & tmask]; uint8_t chk = (uint8_t)(key >> 56);
                if (s->chk != chk) { s->chk = chk; s->p = 32768; s->n = 0; }
                sl[ns++] = s; st[k++] = s->n ? stretchf((s->p + 0.5f) / 65536.0f) : 0.0f;
            }
            int mexp[3], midx[3];
            for (int l = 0; l < nlev; l++) {
                mexp[l] = -1; midx[l] = 0; st[k] = 0;
                if (mlen[l] > 0 && !(gate && lev[l] == 0 && !(b <= 3 && (c >> 4) == 3)) && !(!gate && l > 0 && mlen[l - 1] > 0 && mptr[l - 1] == mptr[l])) {
                    int pb = x[mptr[l]];  // predicted byte (exact), read through the level's abstraction
                    int keep = 0xFF;
                    if (lev[l] >= 1 && pb >= '0' && pb <= '9') keep = 0xF0;
                    if (lev[l] >= 2 && isletter(pb)) keep = 0xDF;
                    int known = (keep >> b) & 1;
                    int hi = partial ^ (1 << (7 - b));  // bits already coded
                    int pbh = pb >> (b + 1);
                    int kh = keep >> (b + 1);
                    if (known && ((hi ^ pbh) & kh) == 0) {
                        int e = (pb >> b) & 1, ln = mlen[l] > 31 ? 31 : mlen[l];
                        midx[l] = ln * 2 + e; mexp[l] = e;
                        float pc = (mstat[l][midx[l]][0] + 0.5f) / 65536.0f;
                        st[k] = e ? stretchf(pc) : -stretchf(pc);
                    }
                }
                k++;
            }
            st[k++] = 0.3f;
            int s1 = (mbucket * 8 + (7 - b)) * nin, s2 = ((i ? x[i - 1] : 0) * 8 + (7 - b)) * nin;
            float d1 = 0, d2 = 0; for (int q = 0; q < nin; q++) { d1 += w1[s1 + q] * st[q]; d2 += w2[s2 + q] * st[q]; }
            float p1 = squashf(d1), p2 = squashf(d2), pm = squashf((d1 + d2) * 0.5f);
            if (pm < 1e-5f) pm = 1e-5f; if (pm > 1 - 1e-5f) pm = 1 - 1e-5f;
            float sv = stretchf(pm); if (sv < -7.999f) sv = -7.999f; if (sv > 7.999f) sv = 7.999f;
            float pos = (sv + 8) * 2; int lo = (int)pos; float fr = pos - lo;
            float *ap = apm[((i ? x[i - 1] : 0) << 8) | partial];
            float pa = ap[lo] * (1 - fr) + ap[lo + 1] * fr;
            float pf = 0.25f * pm + 0.75f * pa; if (pf < 1e-5f) pf = 1e-5f; if (pf > 1 - 1e-5f) pf = 1 - 1e-5f;
            double cst = -log2(bit ? pf : 1 - pf); total += cst; cbyte += cst;
            float e1 = bit - p1, e2 = bit - p2;
            for (int q = 0; q < nin; q++) { w1[s1 + q] += 0.0015f * e1 * st[q]; w2[s2 + q] += 0.0015f * e2 * st[q]; }
            ap[lo] += 0.02f * (1 - fr) * (bit - ap[lo]); ap[lo + 1] += 0.02f * fr * (bit - ap[lo + 1]);
            for (int q = 0; q < ns; q++) {
                Slot *s = sl[q];
                int target = bit ? 65535 : 0;
                s->p += (int)((target - (int)s->p) * rate[s->n]);
                if (s->n < LIMIT) s->n++;
            }
            for (int l = 0; l < nlev; l++) if (mexp[l] >= 0) {
                uint16_t *m = &mstat[l][midx[l]][0];
                int ok = bit == mexp[l];
                *m += (int)(((ok ? 65535 : 0) - (int)*m) * 0.02f);
            }
            partial = partial * 2 + bit;
        }
        if (c >= '0' && c <= '9') cdig += cbyte; else cother += cbyte;
        // update match models and word hashes with the coded byte
        for (int l = 0; l < nlev; l++) {
            if (mlen[l] > 0) {
                if (ax[l][mptr[l]] == ax[l][i]) { mlen[l]++; mptr[l]++; if (mptr[l] >= i + 1) mlen[l] = 0; }
                else mlen[l] = 0;
            }
            uint8_t a = ax[l][i];
            if (isletter(a)) word[l] = mix64(word[l] + a + 1);
            else if (word[l]) { pword[l] = word[l]; word[l] = 0; }
        }
    }
    printf("levels=%-3s bytes=%ld  total=%.0f bytes (%.4f bpc)  digits=%.0f  other=%.0f\n",
           argv[2], n, total / 8, total / n, cdig / 8, cother / 8);
    return 0;
}

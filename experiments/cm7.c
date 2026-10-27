// cm7.c — cm2 plus switchable modeling upgrades (feature bitmask), for A/B testing.
//
//   bit 0  F_FINAL  learned final mixer over the per-set mixers (instead of averaging their dot products)
//   bit 1  F_LIN    extra linear-probability input per context (paq8-style), besides the stretched one
//   bit 2  F_TEXT   text contexts on the literal level: sparse (skip-1, skip-1 order-4), column (byte
//                   above + left byte), markup (innermost open bracket kind + word), 3-word context
//   bit 3  F_APM3   third APM keyed by a hashed order-3 context
//   bit 4  F_MIX3   third mixer weight set selected by the previous byte
//   bit 5  F_LONGM  second literal match model with a 24-byte minimum (better candidates)
//   bit 6  F_RUN    run input per context: whether the slot has only ever seen one bit value, and how often
//   bit 7  F_MIX4   fourth mixer set selected by (letters into the current word, class of previous byte, match bucket)
//   bit 8  F_IND    indirect contexts: the bytes that followed the last occurrences of the order-1 / order-2 context
//   bit 9  F_APMM   APM keyed by the match model's state (expected bit, length)
//   bit 10 F_APMW   learned final blend of the mixer output and the APMs (instead of fixed weights)
//   bit 11 F_LRHI   mixer learning-rate floor 0.003 (default 0.002)
//   bit 12 F_LRLO   mixer learning-rate floor 0.0012
//   bit 13 F_MIX5   fifth mixer set selected by the byte two back
//   bit 14 F_LRLL   mixer learning-rate floor 0.0006
//   bit 15 F_LCW    case-folded word contexts (word + previous byte, word bigram), literal level
//   bit 16 F_ST2    gentler bit-history discount on the opposite bit (n -> 2n/3 + 1 instead of n/2)
//   bit 17 F_SP2    two more sparse contexts: bytes 3-4 back, and bytes 1 and 3 back
//   bit 18 F_P2     second, slow direct probability per slot (rate floor 1/255, own hit count) as an extra input
//   bit 19 F_SSE2   chained APM: an order-2 APM refines the order-1 APM's output (instead of the mixer's)
//   bit 20 F_ASSOC  8-way buckets (one 64-byte line) with 16-bit checksums; evict the least-visited slot
//
// Everything else is cm2: bit histories + StateMaps, orders 1-6,8,12,16,24, two word contexts, a match model
// per abstraction level, 2-way buckets, APMs on order 1 and hashed order 2. Ideal code length.
//
// usage: cm3 <file> <levels e.g. 0 | 01> <features (decimal bitmask)> [maxbytes]
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef TB
#define TB 27
#endif
#ifndef ORDSET
#define ORDSET 0
#endif
#ifndef SMLIM
#define SMLIM 1023
#endif
#ifndef APMR
#define APMR 0.02f
#endif
#if ORDSET == 1
#define NORD 12
static const int ORD[NORD] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24};
#elif ORDSET == 2
#define NORD 11
static const int ORD[NORD] = {1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32};
#else
#define NORD 10
static const int ORD[NORD] = {1, 2, 3, 4, 5, 6, 8, 12, 16, 24};
#endif
#define MAXO 32
#define NCTX (NORD + 2)
#define NTXT 11
#define MB 22
#define MINLEN 7
#define MAXMM 4
enum { F_FINAL = 1, F_LIN = 2, F_TEXT = 4, F_APM3 = 8, F_MIX3 = 16, F_LONGM = 32, F_RUN = 64,
       F_MIX4 = 128, F_IND = 256, F_APMM = 512, F_APMW = 1024, F_LRHI = 2048, F_LRLO = 4096,
       F_MIX5 = 8192, F_LRLL = 16384, F_LCW = 32768, F_ST2 = 65536, F_SP2 = 131072, F_P2 = 262144, F_SSE2 = 524288, F_ASSOC = 1048576 };
static int gentle;

typedef struct { uint16_t p; uint8_t st, nq; uint16_t q, chk; } Slot;
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
static uint8_t st_next(uint8_t s, int y) {
    int n0 = s & 15, n1 = s >> 4;
    if (gentle) {
        if (y) { if (n1 < 15) n1++; if (n0 > 3) n0 = n0 * 2 / 3 + 1; }
        else { if (n0 < 15) n0++; if (n1 > 3) n1 = n1 * 2 / 3 + 1; }
    } else if (y) { if (n1 < 15) n1++; if (n0 > 2) n0 = (n0 + 1) / 2; }
    else { if (n0 < 15) n0++; if (n1 > 2) n1 = (n1 + 1) / 2; }
    return (uint8_t)(n0 | n1 << 4);
}
static float rate[1024];
typedef struct { float p; int n; } SM;

static float apm_get(float *ap, float pr, int *lo, float *fr) {
    float sv = stretchf(pr); if (sv < -7.999f) sv = -7.999f; if (sv > 7.999f) sv = 7.999f;
    float pos = (sv + 8) * 2; *lo = (int)pos; *fr = pos - *lo;
    return ap[*lo] * (1 - *fr) + ap[*lo + 1] * *fr;
}
static void apm_upd(float *ap, int lo, float fr, int y) {
    ap[lo] += APMR * (1 - fr) * (y - ap[lo]); ap[lo + 1] += APMR * fr * (y - ap[lo + 1]);
}
static float clampp(float p) { return p < 1e-5f ? 1e-5f : p > 1 - 1e-5f ? 1 - 1e-5f : p; }

typedef struct { int l, minlen; uint32_t *tab; long ptr; int len; float stat[64]; } MM;

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: cm7 file levels features [maxbytes]\n"); return 2; }
    FILE *f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    int F = atoi(argv[3]); gentle = !!(F & F_ST2);
    if (argc > 4 && atol(argv[4]) > 0 && atol(argv[4]) < n) n = atol(argv[4]);
    uint8_t *x = malloc(n); if (fread(x, 1, n, f) != (size_t)n) return 1; fclose(f);
    int lev[3], nlev = 0; for (const char *c = argv[2]; *c; c++) lev[nlev++] = *c - '0';
    for (int i = 0; i < 1024; i++) rate[i] = 1.0f / (i + 1.5f);
    Slot *tab = calloc((size_t)1 << TB, sizeof(Slot));
    uint64_t tmask = ((uint64_t)1 << TB) - 1;
    uint8_t *ax[3];
    for (int l = 0; l < nlev; l++) { ax[l] = malloc(n); for (long i = 0; i < n; i++) ax[l][i] = ab(x[i], lev[l]); }
    MM mm[MAXMM]; int nmm = 0;
    for (int l = 0; l < nlev; l++) mm[nmm++] = (MM){l, MINLEN, NULL, 0, 0, {0}};
    if (F & F_LONGM) mm[nmm++] = (MM){0, 24, NULL, 0, 0, {0}};
    for (int q = 0; q < nmm; q++) { mm[q].tab = calloc((size_t)1 << MB, 4); for (int k = 0; k < 64; k++) mm[q].stat[k] = 0.75f; }
    int tsel[NTXT], ntxt = 0;
    for (int r = 0; r < 5; r++) if (F & F_TEXT) tsel[ntxt++] = r;
    for (int r = 5; r < 7; r++) if (F & F_IND) tsel[ntxt++] = r;
    for (int r = 7; r < 9; r++) if (F & F_LCW) tsel[ntxt++] = r;
    for (int r = 9; r < 11; r++) if (F & F_SP2) tsel[ntxt++] = r;
    int nctx = 1 + nlev * NCTX + ntxt;
    SM *sm = malloc(sizeof(SM) * nctx * 256);
    for (int i = 0; i < nctx * 256; i++) {
        int n0 = (i & 255) & 15, n1 = (i & 255) >> 4;
        sm[i].p = (n1 + 0.4f) / (n0 + n1 + 0.8f); sm[i].n = 0;
    }
    int per = 2 + !!(F & F_LIN) + !!(F & F_RUN) + !!(F & F_P2);
    int nin = per * nctx + nmm + 1;
    int NS1 = 4 * 8, NS2 = (nctx + 1) * 8, NS3 = 256 * 8, NS4 = 256 * 8, NS5 = 256 * 8;
    float *w1 = malloc(sizeof(float) * nin * NS1), *w2 = malloc(sizeof(float) * nin * NS2), *w3 = malloc(sizeof(float) * nin * NS3), *w4 = malloc(sizeof(float) * nin * NS4), *w5 = malloc(sizeof(float) * nin * NS5);
    for (int i = 0; i < nin * NS1; i++) w1[i] = 0.1f;
    for (int i = 0; i < nin * NS2; i++) w2[i] = 0.1f;
    for (int i = 0; i < nin * NS3; i++) w3[i] = 0.1f;
    for (int i = 0; i < nin * NS4; i++) w4[i] = 0.1f;
    for (int i = 0; i < nin * NS5; i++) w5[i] = 0.1f;
    int nmix = 2 + !!(F & F_MIX3) + !!(F & F_MIX4) + !!(F & F_MIX5), i3 = 2, i4 = 2 + !!(F & F_MIX3), i5 = i4 + !!(F & F_MIX4);
    float wf[8 * 256][5]; for (int i = 0; i < 8 * 256; i++) for (int k = 0; k < 5; k++) wf[i][k] = k < nmix ? 1.0f / nmix : 0;
    float (*apm1)[33] = malloc(sizeof(float) * 33 * 65536), (*apm2)[33] = malloc(sizeof(float) * 33 * 65536),
          (*apm3)[33] = malloc(sizeof(float) * 33 * 65536), (*apm4)[33] = malloc(sizeof(float) * 33 * 65536);
    for (int c = 0; c < 65536; c++) for (int k = 0; k < 33; k++) apm1[c][k] = apm2[c][k] = apm3[c][k] = apm4[c][k] = squashf((k - 16) * 0.5f);
    float wa[8][6]; for (int r = 0; r < 8; r++) for (int k = 0; k < 6; k++) wa[r][k] = 0.2f;
    uint16_t *ind1 = calloc(256, 2), *ind2 = calloc(65536, 2);
    float lrfloor = (F & F_LRHI) ? 0.003f : (F & F_LRLL) ? 0.0006f : (F & F_LRLO) ? 0.0012f : 0.002f;
    double total = 0, cdig = 0, cother = 0;
    uint64_t base[3][NCTX]; int uniq[3][NCTX]; uint64_t tbase[NTXT];
    uint64_t word[3] = {0}, pword[3] = {0}, ppword = 0, lword = 0, plword = 0;
    long linestart = 0, prevline = 0; uint8_t openk = 0; int mlen0 = 0, wlen = 0;
    for (long i = 0; i < n; i++) {
        if ((i & ((1L << 24) - 1)) == 0 && i) fprintf(stderr, "%s F=%d %ld MB  %.0f bytes so far (%.4f bpc)\n", argv[2], F, i >> 20, total / 8, total / i);
        for (int l = 0; l < nlev; l++) {
            uint64_t h = 0x9876 + lev[l] * 0x1111; int j = 0, df = l == 0;
            for (int o = 1; o <= MAXO; o++) {
                h = mix64(h + (i - o >= 0 ? ax[l][i - o] : 256) + 1);
                if (!df && i - o >= 0 && ax[l][i - o] != ax[l - 1][i - o]) df = 1;
                if (j < NORD && ORD[j] == o) { base[l][j] = h + j; uniq[l][j] = df; j++; }
            }
            int dw = l == 0;
            for (long q = i - 1; !dw && q >= 0 && q > i - 40; q--) { if (ax[l][q] != ax[l - 1][q]) dw = 1; if (!isletter(x[q]) && !(x[q] >= '0' && x[q] <= '9')) break; }
            base[l][NORD] = mix64(word[l] * 31 + (i ? ax[l][i - 1] : 0) + lev[l] * 777 + 11);
            base[l][NORD + 1] = mix64(word[l] + pword[l] * 0x2545F4914F6CDD1DULL + lev[l] * 999 + 12);
            uniq[l][NORD] = uniq[l][NORD + 1] = dw || uniq[l][1];
        }
        if (ntxt) {
            int b1 = i > 0 ? x[i - 1] : 0, b2 = i > 1 ? x[i - 2] : 0, b3 = i > 2 ? x[i - 3] : 0, b4 = i > 3 ? x[i - 4] : 0, b5 = i > 4 ? x[i - 5] : 0;
            long col = i - linestart; int above = prevline + col < linestart ? x[prevline + col] : 0;
            tbase[0] = mix64(0xA1000000ULL + (b2 << 8 | b3));
            tbase[1] = mix64(0xA2000000ULL + ((uint64_t)b2 << 24 | b3 << 16 | b4 << 8 | b5));
            tbase[2] = mix64(0xA3000000ULL + (above << 8 | b1) + ((uint64_t)(col < 255 ? col : 255) << 20) * 0);
            tbase[3] = mix64(0xA4000000ULL + openk + word[0] * 0x9E3779B97F4A7C15ULL);
            tbase[4] = mix64(0xA5000000ULL + word[0] + pword[0] * 0x2545F4914F6CDD1DULL + ppword * 0x9E3779B97F4A7C15ULL);
            tbase[5] = mix64(0xA6000000ULL + ((uint64_t)ind1[b1] << 8 | b1));
            tbase[6] = mix64(0xA7000000ULL + ((uint64_t)ind2[b2 << 8 | b1] << 16 | b2 << 8 | b1));
            tbase[7] = mix64(0xA8000000ULL + lword * 31 + (isletter(b1) ? (b1 | 32) : b1));
            tbase[8] = mix64(0xA9000000ULL + lword + plword * 0x2545F4914F6CDD1DULL);
            tbase[9] = mix64(0xAA000000ULL + (b3 << 8 | b4));
            tbase[10] = mix64(0xAB000000ULL + (b1 << 8 | b3));
        }
        for (int q = 0; q < nmm; q++) {
            MM *m = &mm[q]; uint8_t *a = ax[m->l];
            if (i >= m->minlen) {
                uint64_t hk = 0; for (int r = 1; r <= m->minlen; r++) hk = hk * 263 + a[i - r];
                hk = mix64(hk + m->minlen) & (((uint64_t)1 << MB) - 1);
                if (m->len == 0) {
                    long cand = m->tab[hk];
                    if (cand > 0) { int len = 0; while (len < 64 && cand - 1 - len >= 0 && a[cand - 1 - len] == a[i - 1 - len]) len++; if (len >= m->minlen) { m->ptr = cand; m->len = len; } }
                }
                m->tab[hk] = (uint32_t)i;
            }
        }
        mlen0 = mm[0].len;
        int c = x[i], partial = 1; double cbyte = 0;
        int mb = mlen0 == 0 ? 0 : mlen0 < 16 ? 1 : mlen0 < 32 ? 2 : 3;
        for (int b = 7; b >= 0; b--) {
            int bit = (c >> b) & 1;
            float st[4 * (1 + 3 * NCTX + NTXT) + MAXMM + 2]; Slot *sl[1 + 3 * NCTX + NTXT]; int smi[1 + 3 * NCTX + NTXT];
            int k = 0, ns = 0, known = 0;
            for (int t = 0; t < nctx; t++) {
                Slot *s = NULL; uint64_t bk = 0; int use = 1;
                if (t == 0) s = &tab[mix64(0x55 + partial) & tmask];
                else if (t <= nlev * NCTX) { int l = (t - 1) / NCTX, j = (t - 1) % NCTX; use = uniq[l][j]; bk = base[l][j]; }
                else bk = tbase[tsel[t - 1 - nlev * NCTX]];
                if (t > 0 && use) {
                    uint64_t key = mix64(bk ^ (partial * 0xD6E8FEB86659FD93ULL));
                    if (F & F_ASSOC) {
                        uint16_t chk = (uint16_t)(key >> 48) | 1;
                        Slot *line = &tab[key & tmask & ~(uint64_t)7]; int vic = 0, vn = 1 << 30;
                        for (int r = 0; r < 8; r++) {
                            if (line[r].chk == chk) { s = &line[r]; break; }
                            int nv = line[r].chk ? (line[r].st & 15) + (line[r].st >> 4) + line[r].nq : -1;
                            if (nv < vn) { vn = nv; vic = r; }
                        }
                        if (!s) { s = &line[vic]; s->chk = chk; s->st = 0; s->p = 32768; s->q = 32768; s->nq = 0; }
                    } else {
                    uint16_t chk = (uint8_t)(key >> 56) | 1;
                    Slot *a = &tab[key & tmask], *bb = &tab[(key ^ 1) & tmask];
                    if (a->chk == chk) s = a; else if (bb->chk == chk) s = bb;
                    else {
                        int na = (a->st & 15) + (a->st >> 4), nb = (bb->st & 15) + (bb->st >> 4);
                        s = na <= nb ? a : bb; s->chk = chk; s->st = 0; s->p = 32768; s->q = 32768; s->nq = 0;
                    }
                    }
                }
                if (!s) { for (int r = 0; r < per; r++) st[k++] = 0; smi[ns] = -1; sl[ns++] = NULL; continue; }
                int si = t * 256 + s->st;
                float ps = sm[si].p < 0.0001f ? 0.0001f : sm[si].p > 0.9999f ? 0.9999f : sm[si].p;
                st[k++] = stretchf(ps);
                st[k++] = s->st ? stretchf((s->p + 0.5f) / 65536.0f) : 0;
                if (F & F_LIN) st[k++] = s->st ? (ps - 0.5f) * 4 : 0;
                if (F & F_RUN) {
                    int n0 = s->st & 15, n1 = s->st >> 4;
                    st[k++] = (n0 == 0 && n1 > 0) ? logf(1 + n1) : (n1 == 0 && n0 > 0) ? -logf(1 + n0) : 0;
                }
                if (F & F_P2) st[k++] = s->nq ? stretchf((s->q + 0.5f) / 65536.0f) : 0;
                if (s->st) known++;
                smi[ns] = si; sl[ns++] = s;
            }
            int mexp[MAXMM], midx[MAXMM];
            for (int q = 0; q < nmm; q++) {
                MM *m = &mm[q]; mexp[q] = -1; midx[q] = 0; st[k] = 0;
                int dup = 0; for (int r = 0; r < q; r++) if (mm[r].len > 0 && mm[r].ptr == m->ptr) dup = 1;
                if (m->len > 0 && !dup) {
                    int pb = x[m->ptr], keep = 0xFF, lv = lev[m->l];
                    if (lv >= 1 && pb >= '0' && pb <= '9') keep = 0xF0;
                    if (lv >= 2 && isletter(pb)) keep = 0xDF;
                    int hi = partial ^ (1 << (7 - b));
                    if (((keep >> b) & 1) && ((hi ^ (pb >> (b + 1))) & (keep >> (b + 1))) == 0) {
                        int e = (pb >> b) & 1, ln = m->len > 31 ? 31 : m->len;
                        midx[q] = ln * 2 + e; mexp[q] = e;
                        float pc = m->stat[midx[q]];
                        st[k] = e ? stretchf(pc) : -stretchf(pc);
                    }
                }
                k++;
            }
            st[k++] = 0.3f;
            int s1 = (mb * 8 + (7 - b)) * nin, s2 = ((known > nctx ? nctx : known) * 8 + (7 - b)) * nin;
            int s3 = (((i ? x[i - 1] : 0)) * 8 + (7 - b)) * nin;
            int cls = i ? (isletter(x[i - 1]) ? 1 + (x[i - 1] < 'a') : (x[i - 1] >= '0' && x[i - 1] <= '9') ? 3 : x[i - 1] == ' ' ? 4 : x[i - 1] < 32 ? 5 : x[i - 1] < 128 ? 6 : 7) : 0;
            int s4 = ((((wlen > 7 ? 7 : wlen) * 8 + cls) * 4 + mb) * 8 + (7 - b)) * nin;
            float d[5] = {0, 0, 0, 0, 0};
            int s5 = ((i > 1 ? x[i - 2] : 0) * 8 + (7 - b)) * nin;
            for (int q = 0; q < nin; q++) { d[0] += w1[s1 + q] * st[q]; d[1] += w2[s2 + q] * st[q]; }
            if (F & F_MIX3) for (int q = 0; q < nin; q++) d[i3] += w3[s3 + q] * st[q];
            if (F & F_MIX4) for (int q = 0; q < nin; q++) d[i4] += w4[s4 + q] * st[q];
            if (F & F_MIX5) for (int q = 0; q < nin; q++) d[i5] += w5[s5 + q] * st[q];
            for (int r = 0; r < nmix; r++) { if (d[r] > 20) d[r] = 20; if (d[r] < -20) d[r] = -20; }
            float pr[5]; for (int r = 0; r < nmix; r++) pr[r] = squashf(d[r]);
            int fsel = ((7 - b) << 8) | (partial & 255);
            float pm;
            if (F & F_FINAL) { float df = 0; for (int r = 0; r < nmix; r++) df += wf[fsel][r] * d[r]; pm = squashf(df); }
            else { float df = 0; for (int r = 0; r < nmix; r++) df += d[r]; pm = squashf(df / nmix); }
            pm = clampp(pm);
            int c1 = ((i ? x[i - 1] : 0) << 8) | partial;
            int c2 = (int)((mix64(((uint64_t)(i > 1 ? x[i - 2] : 0) << 8 | (i ? x[i - 1] : 0)) * 256 + partial)) & 65535);
            int c3 = (int)((mix64(((uint64_t)(i > 2 ? x[i - 3] : 0) << 16 | (i > 1 ? x[i - 2] : 0) << 8 | (i ? x[i - 1] : 0)) * 256 + partial + 0x777)) & 65535);
            int lo1, lo2, lo3 = 0; float fr1, fr2, fr3 = 0, a3 = 0;
            float a1 = apm_get(apm1[c1], pm, &lo1, &fr1), a2 = apm_get(apm2[c2], (F & F_SSE2) ? clampp(a1) : pm, &lo2, &fr2);
            float pf;
            int ms = 0;  // match state for the APM: 0 none, else 1 + expected bit + 2 * min(len, 15)
            if (mexp[0] >= 0) ms = 1 + mexp[0] + 2 * (mm[0].len > 15 ? 15 : mm[0].len);
            int c4 = (ms * 256 + partial) & 65535, lo4 = 0; float fr4 = 0, a4 = 0;
            if (F & F_APM3) a3 = apm_get(apm3[c3], pm, &lo3, &fr3);
            if (F & F_APMM) a4 = apm_get(apm4[c4], pm, &lo4, &fr4);
            float av[6]; int na = 0; av[na++] = pm; av[na++] = a1; av[na++] = a2;
            if (F & F_APM3) av[na++] = a3;
            if (F & F_APMM) av[na++] = a4;
            float sa[6]; for (int r = 0; r < na; r++) sa[r] = stretchf(clampp(av[r]));
            if (F & F_APMW) { float t = 0; for (int r = 0; r < na; r++) t += wa[7 - b][r] * sa[r]; pf = squashf(t); }
            else if (na == 3) pf = 0.3f * pm + 0.35f * a1 + 0.35f * a2;
            else { pf = 0; for (int r = 0; r < na; r++) pf += av[r]; pf /= na; }
            pf = clampp(pf);
            double cst = -log2(bit ? pf : 1 - pf); total += cst; cbyte += cst;
            float lr = lrfloor + 0.02f / (1.0f + i / 200000.0f);
            float e[5]; for (int r = 0; r < nmix; r++) e[r] = bit - pr[r];
            for (int q = 0; q < nin; q++) { w1[s1 + q] += lr * e[0] * st[q]; w2[s2 + q] += lr * e[1] * st[q]; }
            if (F & F_MIX3) for (int q = 0; q < nin; q++) w3[s3 + q] += lr * e[i3] * st[q];
            if (F & F_MIX4) for (int q = 0; q < nin; q++) w4[s4 + q] += lr * e[i4] * st[q];
            if (F & F_MIX5) for (int q = 0; q < nin; q++) w5[s5 + q] += lr * e[i5] * st[q];
            if (F & F_APMW) { float ea = bit - pf; for (int r = 0; r < na; r++) wa[7 - b][r] += 0.0005f * ea * sa[r]; }
            if (F & F_FINAL) { float ef = bit - pm; for (int r = 0; r < nmix; r++) wf[fsel][r] += 0.0005f * ef * d[r]; }
            apm_upd(apm1[c1], lo1, fr1, bit); apm_upd(apm2[c2], lo2, fr2, bit);
            if (F & F_APM3) apm_upd(apm3[c3], lo3, fr3, bit);
            if (F & F_APMM) apm_upd(apm4[c4], lo4, fr4, bit);
            for (int q = 0; q < ns; q++) {
                Slot *s = sl[q]; if (!s) continue;
                SM *m = &sm[smi[q]];
                m->p += (bit - m->p) * rate[m->n]; if (m->n < SMLIM) m->n++;
                int cnt = (s->st & 15) + (s->st >> 4);
                s->p += (int)(((bit ? 65535 : 0) - (int)s->p) * rate[cnt < 30 ? cnt : 30]);
                s->st = st_next(s->st, bit);
                if (F & F_P2) { s->q += (int)(((bit ? 65535 : 0) - (int)s->q) * rate[s->nq]); if (s->nq < 254) s->nq++; }
            }
            for (int q = 0; q < nmm; q++) if (mexp[q] >= 0) mm[q].stat[midx[q]] += 0.01f * ((bit == mexp[q]) - mm[q].stat[midx[q]]);
            partial = partial * 2 + bit;
        }
        if (c >= '0' && c <= '9') cdig += cbyte; else cother += cbyte;
        for (int q = 0; q < nmm; q++) {
            MM *m = &mm[q];
            if (m->len > 0) { if (ax[m->l][m->ptr] == ax[m->l][i]) { m->len++; m->ptr++; } else m->len = 0; }
        }
        for (int l = 0; l < nlev; l++) {
            uint8_t a = ax[l][i];
            if (isletter(a)) word[l] = mix64(word[l] + a + 1);
            else if (word[l]) { if (l == 0) ppword = pword[0]; pword[l] = word[l]; word[l] = 0; }
        }
        if (i > 0) ind1[x[i - 1]] = (uint16_t)(ind1[x[i - 1]] << 8 | c);
        if (i > 1) ind2[x[i - 2] << 8 | x[i - 1]] = (uint16_t)(ind2[x[i - 2] << 8 | x[i - 1]] << 8 | c);
        wlen = isletter(c) ? wlen + 1 : 0;
        if (isletter(c)) lword = mix64(lword + (c | 32) + 1); else if (lword) { plword = lword; lword = 0; }
        if (c == '\n') { prevline = linestart; linestart = i + 1; }
        if (c == '[' || c == '{' || c == '<' || c == '(' || c == '|' || c == '=') openk = (uint8_t)c;
        else if (c == ']' || c == '}' || c == '>' || c == ')' || c == '\n') openk = 0;
    }
    printf("levels=%-3s F=%-3d bytes=%ld  total=%.0f bytes (%.4f bpc)  digits=%.0f  other=%.0f\n",
           argv[2], F, n, total / 8, total / n, cdig / 8, cother / 8);
    return 0;
}

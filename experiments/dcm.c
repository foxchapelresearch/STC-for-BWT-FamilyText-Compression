// dcm.c — ideal code length of the digits of a text, coded *after* the digit-masked skeleton.
//
// Decoder-side information available when coding run r = [a,e) at digit j:
//   the whole skeleton m (digits -> '9'), all earlier runs (document order), digits a..a+j-1.
// Each digit is binarized MSB-first (4 decisions; after bit3=1 the next two are forced 0).
// Context models -> logistic mixer -> APM, as in PAQ-family coders. Output is sum(-log2 p).
//
// usage: dcm <file> <model-list e.g. 0,1,2> [limit] [lr] [percost_out]
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NMODEL 16
#define TBITS 22
#define CAP 12

typedef struct { float p; uint16_t n; uint16_t chk; } Slot;

static uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
static uint64_t H(uint64_t h, uint64_t v) { return mix64(h * 0x9E3779B97F4A7C15ULL + v + 1); }
static float squash(float t) { if (t > 30) t = 30; if (t < -30) t = -30; return 1.0f / (1.0f + expf(-t)); }
static float stretch(float p) { return logf(p / (1.0f - p)); }

static uint8_t *x, *m;
static long n;

static uint64_t left_hash(long a, int k) {  // m[a-k..a-1], boundary-aware
    uint64_t h = 0x1234 + k;
    for (int i = 1; i <= k; i++) { long q = a - i; h = H(h, q >= 0 ? m[q] : 256); }
    return h;
}
static uint64_t right_hash(long e, int k) {  // m[e..e+k-1]
    uint64_t h = 0x5678 + k;
    for (int i = 0; i < k; i++) { long q = e + i; h = H(h, q < n ? m[q] : 256); }
    return h;
}
static uint64_t word_before(long a) {  // last alphabetic word within 24 bytes before the run
    long q = a - 1, lim = a - 24; uint64_t h = 0x9abc;
    while (q >= 0 && q > lim && !((m[q] | 32) >= 'a' && (m[q] | 32) <= 'z')) q--;
    int len = 0;
    while (q >= 0 && q > lim && ((m[q] | 32) >= 'a' && (m[q] | 32) <= 'z') && len < 16) { h = H(h, m[q] | 32); q--; len++; }
    return len ? h : 0;
}

typedef struct { uint64_t key; uint8_t len; uint8_t d[24]; } Field;
#define FBITS 20
static Field *ftab[2];

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: dcm file models [limit] [lr] [percost]\n"); return 2; }
    FILE *f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); n = ftell(f); fseek(f, 0, SEEK_SET);
    x = malloc(n); m = malloc(n); if (fread(x, 1, n, f) != (size_t)n) return 1; fclose(f);
    for (long i = 0; i < n; i++) m[i] = (x[i] >= '0' && x[i] <= '9') ? '9' : x[i];
    int use[NMODEL] = {0}, nm = 0, ids[NMODEL];
    for (char *s = strtok(argv[2], ","); s; s = strtok(NULL, ",")) { int k = atoi(s); use[k] = 1; ids[nm++] = k; }
    int limit = argc > 3 ? atoi(argv[3]) : 255;
    float lr = argc > 4 ? atof(argv[4]) : 0.006f;
    FILE *pc = argc > 5 ? fopen(argv[5], "wb") : NULL;

    Slot *tab[NMODEL];
    for (int i = 0; i < NMODEL; i++) tab[i] = use[i] ? calloc((size_t)1 << TBITS, sizeof(Slot)) : NULL;
    for (int i = 0; i < 2; i++) ftab[i] = calloc((size_t)1 << FBITS, sizeof(Field));
    int nin = nm + 1, nsel = CAP * CAP * 4;
    float *w = malloc(sizeof(float) * nin * nsel);
    for (int i = 0; i < nin * nsel; i++) w[i] = 0.3f;
    // APM over (bucket, pos, partial) x 33 stretch buckets
    int napm = CAP * CAP * 16;
    float *apm = malloc(sizeof(float) * napm * 33);
    for (int c = 0; c < napm; c++) for (int k = 0; k < 33; k++) apm[c * 33 + k] = squash((k - 16) * 0.5f);

    double total = 0, cost_order0 = 0; long ndig = 0, nruns = 0;
    long dcount[10] = {0};
    long article = 0;
    uint8_t glob[3] = {10, 10, 10};  // last digits in document order (across runs)
    long scan = 0;

    for (long a = 0; a < n;) {
        if (m[a] != '9') { a++; continue; }
        long e = a; while (e < n && m[e] == '9') e++;
        int L = (int)(e - a), Lb = L < CAP ? L : CAP;
        for (; scan < a; scan++)  // count "<page>" occurrences before this run (article index)
            if (m[scan] == '<' && scan + 5 < n && !memcmp(m + scan, "<page>", 6)) article++;
        uint64_t l1 = left_hash(a, 1), l3 = left_hash(a, 3), l6 = left_hash(a, 6), l8 = left_hash(a, 8),
                 l12 = left_hash(a, 12), l24 = left_hash(a, 24), r1 = right_hash(e, 1), r2 = right_hash(e, 2),
                 r3 = right_hash(e, 3), r4 = right_hash(e, 4), wb = word_before(a);
        uint64_t fkey[2] = {left_hash(a, 16) | 1, l6 | 1};
        Field *prev[2];
        for (int t = 0; t < 2; t++) {
            Field *fe = &ftab[t][fkey[t] & (((uint64_t)1 << FBITS) - 1)];
            prev[t] = (fe->key == fkey[t]) ? fe : NULL;
        }
        int fstate[2] = {0, 0};  // 0 equal-so-far, 1 greater, 2 less, 3 n/a
        for (int t = 0; t < 2; t++) fstate[t] = prev[t] && prev[t]->len == L ? 0 : 3;
        uint64_t pre = 0x77;
        int last1 = 10, last2 = 10;
        double runcost = 0;
        for (int j = 0; j < L; j++) {
            int d = x[a + j] - '0';
            int jj = j < CAP - 1 ? j : CAP - 1;
            uint64_t base = H(H(0, Lb), jj);
            int pd[2];
            for (int t = 0; t < 2; t++) {
                pd[t] = 10;
                if (prev[t]) { int idx = j - (L - prev[t]->len); if (idx >= 0 && idx < prev[t]->len) pd[t] = prev[t]->d[idx]; }
            }
            uint64_t ctx[NMODEL];
            ctx[0] = H(base, pre);
            ctx[1] = H(H(base, last1), last2 + 100);
            ctx[2] = H(H(base, pre), l1);
            ctx[3] = H(H(base, pre), l3);
            ctx[4] = H(H(base, pre), l6);
            ctx[5] = H(H(base, last1), l12);
            ctx[6] = H(H(base, pre), l24);
            ctx[7] = H(H(base, pre), r1);
            ctx[8] = H(H(base, pre), r4);
            ctx[9] = H(H(H(base, last1), l3), r3);
            ctx[10] = H(H(H(H(base, fstate[0]), pd[0]), fkey[0]), 1000);
            ctx[11] = H(H(H(H(base, fstate[1]), pd[1]), fkey[1]), 2000);
            ctx[12] = H(H(H(H(0, jj), glob[0]), glob[1]), glob[2] + 3000);
            ctx[13] = H(H(base, pre), wb + 4000);
            ctx[14] = H(H(base, pre), article * 7 + 5000);
            ctx[15] = H(H(H(base, pre), l8), r2);
            // binarize: MSB-first over 4 bits; after bit3 == 1 the digit is 8/9 and bits 2,1 are forced 0
            int partial = 1, top = 0;
            for (int b = 3; b >= 0; b--) {
                int bit = (d >> b) & 1;
                if (top && (b == 2 || b == 1)) { partial = partial * 2 + bit; continue; }
                float st[NMODEL + 1]; Slot *sl[NMODEL];
                for (int k = 0; k < nm; k++) {
                    int i = ids[k];
                    uint64_t hh = mix64(ctx[i] ^ ((uint64_t)partial * 0xD6E8FEB86659FD93ULL));
                    Slot *s = &tab[i][hh & (((uint64_t)1 << TBITS) - 1)];
                    uint16_t chk = (uint16_t)(hh >> 48);
                    if (s->chk != chk) { s->chk = chk; s->p = 0.5f; s->n = 0; }
                    sl[k] = s;
                    float p = s->p; if (p < 1e-4f) p = 1e-4f; if (p > 1 - 1e-4f) p = 1 - 1e-4f;
                    st[k] = stretch(p);
                }
                st[nm] = 0.3f;
                int sel = ((Lb - 1) * CAP + jj) * 4 + (fstate[0] < 3 ? fstate[0] : 3);
                float *ww = w + sel * nin, dot = 0;
                for (int k = 0; k < nin; k++) dot += ww[k] * st[k];
                float pm = squash(dot);
                if (pm < 1e-5f) pm = 1e-5f; if (pm > 1 - 1e-5f) pm = 1 - 1e-5f;
                // APM refine
                int ac = ((Lb - 1) * CAP + jj) * 16 + partial;
                float sv = stretch(pm); if (sv < -7.999f) sv = -7.999f; if (sv > 7.999f) sv = 7.999f;
                float pos = (sv + 8.0f) * 2.0f; int lo = (int)pos; float fr = pos - lo;
                float *ap = apm + ac * 33;
                float pa = ap[lo] * (1 - fr) + ap[lo + 1] * fr;
                float p = 0.4f * pm + 0.6f * pa;
                if (p < 1e-5f) p = 1e-5f; if (p > 1 - 1e-5f) p = 1 - 1e-5f;
                double c = -log2(bit ? p : 1 - p);
                total += c; runcost += c;
                // updates
                float err = bit - pm;
                for (int k = 0; k < nin; k++) ww[k] += lr * err * st[k];
                ap[lo] += 0.02f * (1 - fr) * (bit - ap[lo]);
                ap[lo + 1] += 0.02f * fr * (bit - ap[lo + 1]);
                for (int k = 0; k < nm; k++) {
                    Slot *s = sl[k];
                    s->p += (bit - s->p) / (s->n + 1.5f);
                    if (s->n < limit) s->n++;
                }
                partial = partial * 2 + bit;
                if (b == 3) top = bit;
            }
            for (int t = 0; t < 2; t++)
                if (fstate[t] == 0 && pd[t] < 10 && d != pd[t]) fstate[t] = d > pd[t] ? 1 : 2;
            pre = H(pre, d); last2 = last1; last1 = d;
            glob[2] = glob[1]; glob[1] = glob[0]; glob[0] = d;
            dcount[d]++; ndig++;
        }
        if (pc) { float cc = (float)runcost; fwrite(&a, sizeof(long), 1, pc); fwrite(&cc, sizeof(float), 1, pc); }
        for (int t = 0; t < 2; t++) {
            Field *fe = &ftab[t][fkey[t] & (((uint64_t)1 << FBITS) - 1)];
            fe->key = fkey[t]; fe->len = L > 24 ? 24 : L;
            for (int j = 0; j < fe->len; j++) fe->d[j] = x[a + j] - '0';
        }
        nruns++;
        a = e;
    }
    for (int d = 0; d < 10; d++) if (dcount[d]) cost_order0 -= dcount[d] * log2((double)dcount[d] / ndig);
    printf("runs=%ld digits=%ld order0=%.0f bytes (%.3f bpd)  model=%.0f bytes (%.3f bits/digit, %.2f bits/run)\n",
           nruns, ndig, cost_order0 / 8, cost_order0 / ndig, total / 8, total / ndig, total / nruns);
    if (pc) fclose(pc);
    return 0;
}

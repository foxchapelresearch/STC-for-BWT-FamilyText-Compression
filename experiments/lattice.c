// lattice.c — hierarchical shrinkage over a lattice of context abstractions (sequential, ideal code length).
//
// Abstraction levels a: 0 = literal, 1 = digits -> '9', 2 = digits -> '9' + case folded.
// Node (j, a) = the last j bytes passed through abstraction a. Each byte is coded as 8 binary decisions
// (MSB first, the partial byte is part of every node's context).
// A node's prediction shrinks its own counts toward a blend of its parents:
//     suffix parent      (j-1, a)
//     abstraction parent (j, a+1)            if level a+1 is enabled
//     p = (n1 - D[n1>0] + (T + D*u) * q) / (n + T),   q = w*p_suffix + (1-w)*p_abstraction
// where u = number of distinct outcomes seen and w is learned per node type by Bayesian averaging of the
// two parents' past log-loss. "Hard merge" = enable only one level. Optionally a logistic mixer blends
// every node's prediction ("mix, don't select").
//
// usage: lattice <file> <levels e.g. 0 | 1 | 01 | 012> <K> <mix 0|1> [D] [T] [maxbytes]
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TB 26
#define MAXK 8
#define NL 3

typedef struct { uint32_t chk; uint16_t n0, n1; } Slot;

static uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL; return z ^ (z >> 31);
}
static float stretchf(float p) { return logf(p / (1 - p)); }
static float squashf(float t) { if (t > 20) t = 20; if (t < -20) t = -20; return 1 / (1 + expf(-t)); }
static uint8_t ab(uint8_t b, int a) {
    if (a >= 1 && b >= '0' && b <= '9') b = '9';
    if (a >= 2 && b >= 'A' && b <= 'Z') b += 32;
    return b;
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: lattice file levels K mix [D] [T] [maxbytes]\n"); return 2; }
    FILE *f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    const char *lv = argv[2]; int K = atoi(argv[3]), usemix = atoi(argv[4]);
    float D = argc > 5 ? atof(argv[5]) : 0.6f, T = argc > 6 ? atof(argv[6]) : 0.4f;
    if (argc > 7 && atol(argv[7]) > 0 && atol(argv[7]) < n) n = atol(argv[7]);
    uint8_t *x = malloc(n); if (fread(x, 1, n, f) != (size_t)n) return 1; fclose(f);
    int on[NL] = {0}; for (const char *c = lv; *c; c++) on[*c - '0'] = 1;
    int lev[NL], nlev = 0; for (int a = 0; a < NL; a++) if (on[a]) lev[nlev++] = a;
    Slot *tab = calloc((size_t)1 << TB, sizeof(Slot));
    // node index: root = 0; (j, li) = 1 + (j-1)*nlev + li  for j = 1..K, li = index into lev[]
    int nn = 1 + K * nlev;
    double lossS[1 + MAXK * NL] = {0}, lossA[1 + MAXK * NL] = {0};
    int nin = nn + 1; float *w = malloc(sizeof(float) * nin * 8 * (MAXK + 2));
    for (int i = 0; i < nin * 8 * (MAXK + 2); i++) w[i] = 1.0f / nn;
    double total = 0, cdig = 0, cother = 0; long ndig = 0;
    uint64_t h[MAXK + 1][NL]; int diff[MAXK + 1][NL];
    for (long i = 0; i < n; i++) {
        for (int li = 0; li < nlev; li++) {
            uint64_t hh = 0x1234567ULL + lev[li] * 0x9999;
            h[0][li] = 0;
            diff[0][li] = 0;
            for (int j = 1; j <= K; j++) {
                hh = mix64(hh + (i - j >= 0 ? ab(x[i - j], lev[li]) : 256) + 1); h[j][li] = hh;
                // does the next-coarser enabled level see a different window? (else its node duplicates this one)
                diff[j][li] = diff[j - 1][li] || (li + 1 < nlev && i - j >= 0 && ab(x[i - j], lev[li]) != ab(x[i - j], lev[li + 1]));
            }
        }
        int c = x[i], partial = 1; double cbyte = 0;
        for (int b = 7; b >= 0; b--) {
            int bit = (c >> b) & 1;
            float p[1 + MAXK * NL]; Slot *sl[1 + MAXK * NL]; float ps[1 + MAXK * NL], pa[1 + MAXK * NL];
            int hit = 0;
            for (int node = 0; node < nn; node++) {
                uint64_t key;
                if (node == 0) key = mix64(0xABCDEFULL + partial);
                else { int j = 1 + (node - 1) / nlev, li = (node - 1) % nlev; key = mix64(h[j][li] ^ (partial * 0xD6E8FEB86659FD93ULL)); }
                Slot *s = &tab[key & (((uint64_t)1 << TB) - 1)];
                uint32_t chk = (uint32_t)(key >> 32);
                if (s->chk != chk) { s->chk = chk; s->n0 = s->n1 = 0; }
                sl[node] = s;
            }
            // bottom-up: root, then by order j; within an order, coarsest level first
            for (int node = 0; node < nn; node++) ps[node] = pa[node] = -1;
            for (int j = 0; j <= K; j++) {
                for (int li = (j == 0 ? 0 : nlev - 1); li >= 0; li--) {
                    int node = j == 0 ? 0 : 1 + (j - 1) * nlev + li;
                    float q;
                    if (j == 0) q = 0.5f;
                    else {
                        int sn = j == 1 ? 0 : 1 + (j - 2) * nlev + li;
                        float qs = p[sn]; ps[node] = qs;
                        if (li + 1 < nlev && diff[j][li]) {
                            float qa = p[1 + (j - 1) * nlev + li + 1]; pa[node] = qa;
                            double ws = 1.0 / (1.0 + exp2(lossS[node] - lossA[node]));
                            q = (float)(ws * qs + (1 - ws) * qa);
                        } else q = qs;
                    }
                    Slot *s = sl[node]; float nt = s->n0 + s->n1;
                    float pr;
                    if (nt == 0) pr = q;
                    else {
                        int u = (s->n0 > 0) + (s->n1 > 0);
                        pr = (s->n1 - (s->n1 > 0 ? D : 0) + (T + D * u) * q) / (nt + T);
                    }
                    if (pr < 1e-5f) pr = 1e-5f; if (pr > 1 - 1e-5f) pr = 1 - 1e-5f;
                    p[node] = pr;
                    if (nt > 0 && j > hit) hit = j;
                }
            }
            int top = nn - 1 - (nlev - 1);  // (K, finest enabled level)
            float pf = p[top];
            float st[1 + MAXK * NL + 1]; int sel = 0;
            if (usemix) {
                for (int node = 0; node < nn; node++) st[node] = stretchf(p[node]);
                st[nn] = 0.3f;
                sel = ((7 - b) * (MAXK + 2) + hit) * nin;
                float dot = 0; for (int k = 0; k < nin; k++) dot += w[sel + k] * st[k];
                pf = squashf(dot); if (pf < 1e-5f) pf = 1e-5f; if (pf > 1 - 1e-5f) pf = 1 - 1e-5f;
            }
            double cst = -log2(bit ? pf : 1 - pf); total += cst; cbyte += cst;
            if (usemix) { float err = bit - pf; for (int k = 0; k < nin; k++) w[sel + k] += 0.002f * err * st[k]; }
            for (int node = 1; node < nn; node++) if (pa[node] >= 0) {
                lossS[node] = 0.998 * lossS[node] - log2(bit ? ps[node] : 1 - ps[node]);
                lossA[node] = 0.998 * lossA[node] - log2(bit ? pa[node] : 1 - pa[node]);
            }
            for (int node = 0; node < nn; node++) {
                Slot *s = sl[node];
                if (bit) s->n1++; else s->n0++;
                if (s->n0 + s->n1 > 1020) { s->n0 = (s->n0 + 1) / 2; s->n1 = (s->n1 + 1) / 2; }
            }
            partial = partial * 2 + bit;
        }
        if (c >= '0' && c <= '9') { cdig += cbyte; ndig++; } else cother += cbyte;
    }
    printf("levels=%-3s K=%d mix=%d D=%.2f T=%.2f  bytes=%ld  total=%.0f bytes (%.4f bpc)  digits=%.0f  other=%.0f\n",
           lv, K, usemix, D, T, n, total / 8, total / n, cdig / 8, cother / 8);
    return 0;
}

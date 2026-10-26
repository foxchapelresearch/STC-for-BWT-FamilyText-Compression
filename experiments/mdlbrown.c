// mdlbrown.c — automatic MDL-scored merge search over context abstractions, per decision node.
//
// Each byte is coded as 8 binary decisions, MSB first (node d = 0..7; node d sees the d bits already
// coded as its "partial"). Node d predicts its bit from (phi_d(x[i-1]), ..., phi_d(x[i-K]), partial),
// where phi_d is a partition of the byte alphabet, applied to every context position.
//
// Search (agglomerative, starts from the identity partition):
//   propose: the M pairs of classes with the smallest Information Bottleneck merge cost, computed from
//            each class's distribution of node outcomes given it appears at context position 1 or 2;
//   accept : the proposal with the most negative exact change in code length; stop when none is < 0.
// Code length = sequential KT (Dirichlet 1/2) code length of the node's bits, summed over contexts.
//
// usage: mdlbrown <file> <node 0..7 | shared> [M] [maxbytes]
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define K 4
#define NOUT 512

typedef struct { uint64_t key; uint32_t n0, n1; } Ent;  // key = (ctx tuple << 9) | partial ; 0 = empty

static double ktbits(uint32_t n0, uint32_t n1) {
    static double lg05 = 0; if (!lg05) lg05 = lgamma(0.5);
    return (lgamma(n0 + n1 + 1.0) - lgamma(n0 + 0.5) - lgamma(n1 + 0.5) + 2 * lg05) / log(2.0);
}
static double fitbits(uint32_t n0, uint32_t n1) {
    double n = n0 + n1, r = 0;
    if (n0) r -= n0 * log2(n0 / n);
    if (n1) r -= n1 * log2(n1 / n);
    return r;
}
static uint64_t hk(uint64_t k) { k *= 0x9E3779B97F4A7C15ULL; return k ^ (k >> 31); }

typedef struct {  // open-addressing accumulator
    Ent *t; uint64_t cap, used;
} Tab;
static void tab_init(Tab *T, uint64_t want) {
    T->cap = 1; while (T->cap < want * 2) T->cap <<= 1;
    T->t = calloc(T->cap, sizeof(Ent)); T->used = 0;
}
static Ent *tab_get(Tab *T, uint64_t key) {
    uint64_t p = hk(key) & (T->cap - 1);
    while (T->t[p].key && T->t[p].key != key) p = (p + 1) & (T->cap - 1);
    if (!T->t[p].key) { T->t[p].key = key; T->used++; }
    return &T->t[p];
}

typedef struct {
    int d;
    Ent *raw; uint64_t nraw;          // raw byte contexts
    Ent *g; uint64_t ng;              // aggregated under current partition
    uint32_t *idxoff, *idx;           // per-class lists of g entries containing that class
    uint64_t prof[256][2][NOUT];      // outcome counts given byte at context position 1 / 2
    double L, fit;
    uint8_t map[256];                 // byte -> class representative
} Node;

static const uint8_t *X; static long NX;

static uint64_t tuple_of(uint64_t rawkey, const uint8_t *map) {
    uint64_t ctx = (rawkey - 1) >> 9, out = 0;
    for (int p = 0; p < K; p++) out |= (uint64_t)map[(ctx >> (8 * p)) & 255] << (8 * p);
    return out;
}

static void build_raw(Node *N) {
    Tab T; tab_init(&T, 1 << 23);
    int d = N->d;
    memset(N->prof, 0, sizeof N->prof);
    for (long i = K; i < NX; i++) {
        uint8_t b = X[i];
        uint64_t partial = (1u << d) | (b >> (8 - d)), bit = (b >> (7 - d)) & 1;
        uint64_t ctx = 0;
        for (int p = 0; p < K; p++) ctx |= (uint64_t)X[i - 1 - p] << (8 * p);
        Ent *e = tab_get(&T, ((ctx << 9) | partial) + 1);
        if (bit) e->n1++; else e->n0++;
        if (T.used * 10 > T.cap * 7) {  // grow
            Tab U; tab_init(&U, T.cap);
            for (uint64_t q = 0; q < T.cap; q++) if (T.t[q].key) { Ent *f = tab_get(&U, T.t[q].key); f->n0 = T.t[q].n0; f->n1 = T.t[q].n1; }
            free(T.t); T = U;
        }
        int o = (int)(partial * 2 + bit);
        N->prof[X[i - 1]][0][o]++; N->prof[X[i - 2]][1][o]++;
    }
    N->raw = malloc(T.used * sizeof(Ent)); N->nraw = 0;
    for (uint64_t q = 0; q < T.cap; q++) if (T.t[q].key) N->raw[N->nraw++] = T.t[q];
    free(T.t);
}

static void rebuild(Node *N) {
    Tab T; tab_init(&T, N->nraw / 2 + 1024);
    for (uint64_t r = 0; r < N->nraw; r++) {
        uint64_t key = ((tuple_of(N->raw[r].key, N->map) << 9) | (((N->raw[r].key - 1)) & 511)) + 1;
        Ent *e = tab_get(&T, key);
        e->n0 += N->raw[r].n0; e->n1 += N->raw[r].n1;
        if (T.used * 10 > T.cap * 7) {
            Tab U; tab_init(&U, T.cap);
            for (uint64_t q = 0; q < T.cap; q++) if (T.t[q].key) { Ent *f = tab_get(&U, T.t[q].key); f->n0 = T.t[q].n0; f->n1 = T.t[q].n1; }
            free(T.t); T = U;
        }
    }
    free(N->g); N->g = malloc(T.used * sizeof(Ent)); N->ng = 0;
    N->L = 0; N->fit = 0;
    for (uint64_t q = 0; q < T.cap; q++) if (T.t[q].key) {
        N->g[N->ng++] = T.t[q];
        N->L += ktbits(T.t[q].n0, T.t[q].n1); N->fit += fitbits(T.t[q].n0, T.t[q].n1);
    }
    free(T.t);
    // per-class index (an entry is listed once per distinct class it contains)
    free(N->idxoff); free(N->idx);
    N->idxoff = calloc(257, sizeof(uint32_t));
    for (uint64_t e = 0; e < N->ng; e++) {
        uint64_t t = (N->g[e].key - 1) >> 9; uint8_t seen[K]; int ns = 0;
        for (int p = 0; p < K; p++) { uint8_t c = (t >> (8 * p)) & 255; int dup = 0; for (int s = 0; s < ns; s++) dup |= seen[s] == c; if (!dup) { seen[ns++] = c; N->idxoff[c + 1]++; } }
    }
    for (int c = 0; c < 256; c++) N->idxoff[c + 1] += N->idxoff[c];
    N->idx = malloc((N->idxoff[256] + 1) * sizeof(uint32_t));
    uint32_t *fill = malloc(256 * sizeof(uint32_t)); memcpy(fill, N->idxoff, 256 * sizeof(uint32_t));
    for (uint64_t e = 0; e < N->ng; e++) {
        uint64_t t = (N->g[e].key - 1) >> 9; uint8_t seen[K]; int ns = 0;
        for (int p = 0; p < K; p++) { uint8_t c = (t >> (8 * p)) & 255; int dup = 0; for (int s = 0; s < ns; s++) dup |= seen[s] == c; if (!dup) { seen[ns++] = c; N->idx[fill[c]++] = (uint32_t)e; } }
    }
    free(fill);
}

// exact change in code length if class b is merged into class a (a < b)
static uint32_t *stamp; static uint32_t stampv;
static double delta_merge(Node *N, int a, int b, double *dfit) {
    uint32_t na = N->idxoff[a + 1] - N->idxoff[a], nb = N->idxoff[b + 1] - N->idxoff[b];
    Tab T; tab_init(&T, na + nb + 16);
    double old = 0, oldf = 0; stampv++;
    for (int w = 0; w < 2; w++) {
        int c = w ? b : a;
        for (uint32_t q = N->idxoff[c]; q < N->idxoff[c + 1]; q++) {
            uint32_t e = N->idx[q];
            if (stamp[e] == stampv) continue;
            stamp[e] = stampv;
            Ent *g = &N->g[e];
            old += ktbits(g->n0, g->n1); oldf += fitbits(g->n0, g->n1);
            uint64_t t = (g->key - 1) >> 9, nt = 0;
            for (int p = 0; p < K; p++) { uint64_t cc = (t >> (8 * p)) & 255; if ((int)cc == b) cc = a; nt |= cc << (8 * p); }
            Ent *f = tab_get(&T, ((nt << 9) | ((g->key - 1) & 511)) + 1);
            f->n0 += g->n0; f->n1 += g->n1;
        }
    }
    double nw = 0, nwf = 0;
    for (uint64_t q = 0; q < T.cap; q++) if (T.t[q].key) { nw += ktbits(T.t[q].n0, T.t[q].n1); nwf += fitbits(T.t[q].n0, T.t[q].n1); }
    free(T.t);
    *dfit = nwf - oldf;
    return nw - old;
}

static double H(const uint64_t *v, int n, double *tot) {
    double s = 0, h = 0; for (int i = 0; i < n; i++) s += v[i];
    for (int i = 0; i < n; i++) if (v[i]) h -= v[i] * log2(v[i] / s);
    *tot = s; return h;  // returns total * entropy
}
// IB merge cost (bits) of two classes, from their summed outcome profiles at positions 1 and 2
static double ib_cost(uint64_t (*pa)[NOUT], uint64_t (*pb)[NOUT]) {
    double c = 0;
    for (int w = 0; w < 2; w++) {
        uint64_t mix[NOUT]; double ta, tb, tm;
        for (int o = 0; o < NOUT; o++) mix[o] = pa[w][o] + pb[w][o];
        double ha = H(pa[w], NOUT, &ta), hb = H(pb[w], NOUT, &tb), hm = H(mix, NOUT, &tm);
        c += hm - ha - hb;
    }
    return c;
}

static const char *lab(int b) {
    static char buf[8][8]; static int r; r = (r + 1) & 7;
    if (b == ' ') return "SP"; if (b == '\n') return "NL"; if (b == '\t') return "TAB";
    if (b > 32 && b < 127) { buf[r][0] = (char)b; buf[r][1] = 0; return buf[r]; }
    snprintf(buf[r], 8, "x%02X", b); return buf[r];
}
static void print_class(const uint8_t *map, int rep) {
    for (int b = 0; b < 256; b++) if (map[b] == rep) printf("%s", lab(b));
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: mdlbrown file node|shared [M] [maxbytes]\n"); return 2; }
    FILE *f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); NX = ftell(f); fseek(f, 0, SEEK_SET);
    int M = argc > 3 ? atoi(argv[3]) : 60;
    if (argc > 4 && atol(argv[4]) > 0 && atol(argv[4]) < NX) NX = atol(argv[4]);
    uint8_t *x = malloc(NX); if (fread(x, 1, NX, f) != (size_t)NX) return 1; fclose(f); X = x;
    int shared = !strcmp(argv[2], "shared");
    int d0 = shared ? 0 : atoi(argv[2]), d1 = shared ? 7 : d0, nn = d1 - d0 + 1;
    Node *nodes = calloc(nn, sizeof(Node));
    long cnt[256] = {0}; for (long i = 0; i < NX; i++) cnt[x[i]]++;
    uint64_t maxe = 0;
    for (int j = 0; j < nn; j++) {
        Node *N = &nodes[j]; N->d = d0 + j;
        for (int b = 0; b < 256; b++) N->map[b] = (uint8_t)b;
        build_raw(N); rebuild(N);
        if (N->ng > maxe) maxe = N->ng;
    }
    stamp = calloc(maxe * 4 + 16, sizeof(uint32_t));
    double L0 = 0, F0 = 0; for (int j = 0; j < nn; j++) { L0 += nodes[j].L; F0 += nodes[j].fit; }
    printf("nodes %d..%d  K=%d  bytes=%ld  start: code length %.0f bytes, fit %.0f bytes\n", d0, d1, K, NX, L0 / 8, F0 / 8);

    // current classes: representatives = bytes that occur, each mapping to itself initially
    int alive[256], na = 0; for (int b = 0; b < 256; b++) if (cnt[b]) alive[na++] = b;
    uint64_t (*cp)[2][NOUT] = calloc(256, sizeof *cp);  // class profiles, summed over the searched nodes
    for (int b = 0; b < 256; b++) for (int j = 0; j < nn; j++) for (int w = 0; w < 2; w++) for (int o = 0; o < NOUT; o++) cp[b][w][o] += nodes[j].prof[b][w][o];
    uint8_t *map = nodes[0].map;  // all nodes share the same map in shared mode; per-node runs have one node
    int step = 0; double dfit_total = 0;
    typedef struct { double c; int a, b; } Cand;
    Cand *cand = malloc(sizeof(Cand) * 256 * 256);
    for (;;) {
        int nc = 0;
        for (int i = 0; i < na; i++) for (int j = i + 1; j < na; j++) {
            cand[nc].c = ib_cost(cp[alive[i]], cp[alive[j]]); cand[nc].a = alive[i]; cand[nc].b = alive[j]; nc++;
        }
        // partial selection of the M cheapest proposals
        for (int i = 0; i < M && i < nc; i++) {
            int best = i; for (int j = i + 1; j < nc; j++) if (cand[j].c < cand[best].c) best = j;
            Cand t = cand[i]; cand[i] = cand[best]; cand[best] = t;
        }
        double bestd = 0, bestf = 0; int ba = -1, bb = -1, brank = -1;
        for (int i = 0; i < M && i < nc; i++) {
            double dl = 0, df = 0;
            for (int j = 0; j < nn; j++) { double f1; dl += delta_merge(&nodes[j], cand[i].a, cand[i].b, &f1); df += f1; }
            if (dl < bestd) { bestd = dl; bestf = df; ba = cand[i].a; bb = cand[i].b; brank = i; }
        }
        if (ba < 0) break;
        step++;
        printf("%3d. ΔL %+9.0f bytes (fit %+9.0f)  proposal #%-2d  [", step, bestd / 8, bestf / 8, brank + 1);
        print_class(map, ba); printf("] + ["); print_class(map, bb); printf("]\n");
        fflush(stdout);
        dfit_total += bestf;
        for (int j = 0; j < nn; j++) {
            for (int b = 0; b < 256; b++) if (nodes[j].map[b] == bb) nodes[j].map[b] = (uint8_t)ba;
            rebuild(&nodes[j]);
        }
        for (int w = 0; w < 2; w++) for (int o = 0; o < NOUT; o++) cp[ba][w][o] += cp[bb][w][o];
        int k2 = 0; for (int i = 0; i < na; i++) if (alive[i] != bb) alive[k2++] = alive[i]; na = k2;
    }
    double L1 = 0, F1 = 0; for (int j = 0; j < nn; j++) { L1 += nodes[j].L; F1 += nodes[j].fit; }
    printf("final: code length %.0f bytes (%+.0f), fit %.0f bytes (%+.0f), %d merges, %d classes\n",
           L1 / 8, (L1 - L0) / 8, F1 / 8, (F1 - F0) / 8, step, na);
    printf("multi-member classes:\n");
    for (int i = 0; i < na; i++) {
        int m = 0; for (int b = 0; b < 256; b++) m += map[b] == alive[i];
        if (m > 1) { printf("  {"); print_class(map, alive[i]); printf("}\n"); }
    }
    return 0;
}

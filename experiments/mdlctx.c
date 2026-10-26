// mdlctx.c — fit vs. MDL code length of an order-k model whose context symbols pass through an abstraction phi.
//
// For each position i, the model predicts a target from ctx = (phi(x[i-k]), ..., phi(x[i-1])).
//   node X: target = x[i] (full byte, A = 256)
//   node C: target = class(x[i]) where all digits are one class (A = 247)
//   node M: target = digit value, only at digit positions (A = 10)
// phi: n (none) or flags: d digits->'9', c case fold, u bytes>=0x80->0x80, U capitals->'A',
//      v vowels->'a', p .,;:!?->'.', w newline/tab->space
//
// Reports
//   fit = N * H_hat(target | ctx)  — the maximum-likelihood objective (what Brown clustering maximizes)
//   kt  = sequential KT (Dirichlet 1/2) code length — a proper MDL code: fit + learning cost
//
// usage: mdlctx <file> <X|C|M> <phi> <k>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint64_t key; uint64_t n; } Ent;

static int cmp(const void *a, const void *b) {
    uint64_t x = ((const Ent *)a)->key, y = ((const Ent *)b)->key;
    return x < y ? -1 : x > y;
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: mdlctx file X|C|M phi k\n"); return 2; }
    FILE *f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *x = malloc(n), *c = malloc(n); if (fread(x, 1, n, f) != (size_t)n) return 1; fclose(f);
    char node = argv[2][0]; const char *phi = argv[3]; int k = atoi(argv[4]);
    int pd = strchr(phi, 'd') != NULL, pc = strchr(phi, 'c') != NULL, pu = strchr(phi, 'u') != NULL;
    int pU = strchr(phi, 'U') != NULL, pv = strchr(phi, 'v') != NULL, pp = strchr(phi, 'p') != NULL, pw = strchr(phi, 'w') != NULL;
    for (long i = 0; i < n; i++) {
        uint8_t b = x[i];
        if (pd && b >= '0' && b <= '9') b = '9';
        if (pc && b >= 'A' && b <= 'Z') b += 32;
        if (pu && b >= 0x80) b = 0x80;
        if (pU && b >= 'A' && b <= 'Z') b = 'A';
        if (pv && strchr("aeiou", b) && b) b = 'a';
        if (pp && strchr(".,;:!?", b) && b) b = '.';
        if (pw && (b == '\n' || b == '\t')) b = ' ';
        c[i] = b;
    }
    int A = node == 'X' ? 256 : node == 'C' ? 247 : 10;
    int tb = 40 - 8 * k > 0 ? 0 : 0; (void)tb;
    uint64_t cap = (uint64_t)1 << (k >= 5 ? 27 : k >= 3 ? 25 : 20);
    Ent *t = calloc(cap, sizeof(Ent));
    uint64_t used = 0, N = 0;
    for (long i = k; i < n; i++) {
        int tg;
        uint8_t b = x[i];
        if (node == 'X') tg = b;
        else if (node == 'C') tg = (b >= '0' && b <= '9') ? 256 : b;
        else { if (b < '0' || b > '9') continue; tg = b - '0'; }
        uint64_t ctx = 0;
        for (int j = 1; j <= k; j++) ctx = (ctx << 8) | c[i - j];
        uint64_t key = ((ctx << 9) | (uint64_t)tg) + 1;
        uint64_t h = key * 0x9E3779B97F4A7C15ULL; h ^= h >> 29;
        uint64_t p = h & (cap - 1);
        while (t[p].key && t[p].key != key) p = (p + 1) & (cap - 1);
        if (!t[p].key) { t[p].key = key; if (++used * 10 > cap * 8) { fprintf(stderr, "table full\n"); return 1; } }
        t[p].n++; N++;
    }
    uint64_t m = 0;
    for (uint64_t p = 0; p < cap; p++) if (t[p].key) t[m++] = t[p];
    qsort(t, m, sizeof(Ent), cmp);
    double fit = 0, kt = 0; uint64_t nctx = 0;
    const double a = 0.5, lga = lgamma(a), lgA = lgamma(A * a);
    for (uint64_t i = 0; i < m;) {
        uint64_t ctx = (t[i].key - 1) >> 9, ns = 0; double s1 = 0, s2 = 0;
        uint64_t j = i;
        for (; j < m && ((t[j].key - 1) >> 9) == ctx; j++) {
            double v = (double)t[j].n; ns += t[j].n;
            s1 += v * log2(v);
            s2 += lgamma(v + a) - lga;
        }
        fit += (double)ns * log2((double)ns) - s1;
        kt += (lgamma((double)ns + A * a) - lgA - s2) / log(2.0);
        nctx++; i = j;
    }
    printf("node=%c phi=%-3s k=%d  N=%llu contexts=%llu  fit=%.0f bytes  kt=%.0f bytes  (learning cost %.0f)\n",
           node, phi, k, (unsigned long long)N, (unsigned long long)nctx, fit / 8, kt / 8, (kt - fit) / 8);
    return 0;
}

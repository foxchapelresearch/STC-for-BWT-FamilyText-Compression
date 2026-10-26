// casecm.c — ideal code length of letter case, coded *after* the case-folded skeleton.
//
// Skeleton f = fold(s) where s is the input (optionally digit-masked first) and fold maps A-Z -> a-z.
// For each ASCII word w = f[a..e) the decoder already knows f entirely; we code the case pattern:
//   decision A: all-lower?  B: Capitalized (first upper, rest lower)?  C: ALL-CAPS?  else per-letter bits.
// Context models -> logistic mixer -> APM. Output is sum(-log2 p) in bytes.
//
// usage: casecm <file> <maskdigits 0|1> [folded_out]
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NM 13
#define TBITS 22

typedef struct { float p; uint16_t n; uint16_t chk; } Slot;
static uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
static uint64_t H(uint64_t h, uint64_t v) { return mix64(h * 0x9E3779B97F4A7C15ULL + v + 1); }
static float squash(float t) { if (t > 30) t = 30; if (t < -30) t = -30; return 1.0f / (1.0f + expf(-t)); }
static float stretch(float p) { return logf(p / (1.0f - p)); }

static Slot *tab[NM];
static float w[64][NM + 1];
static float apm[64 * 33];
static double total, tally[8]; static int cur_t;
static long ndec;

static uint8_t *seen;
static int seen_word(uint64_t h) { uint64_t i = h & ((1u << 26) - 1); int r = (seen[i >> 3] >> (i & 7)) & 1; seen[i >> 3] |= 1 << (i & 7); return r; }
static int isup(int c) { return c >= 'A' && c <= 'Z'; }
static int islow(int c) { return c >= 'a' && c <= 'z'; }

// code one binary decision under contexts ctx[0..NM-1], mixer/APM selector sel (<64)
static void code_bit(int bit, const uint64_t *ctx, int sel) {
    float st[NM + 1]; Slot *sl[NM];
    for (int i = 0; i < NM; i++) {
        uint64_t hh = mix64(ctx[i] + i * 0x1234567ULL);
        Slot *s = &tab[i][hh & (((uint64_t)1 << TBITS) - 1)];
        uint16_t chk = (uint16_t)(hh >> 48);
        if (s->chk != chk) { s->chk = chk; s->p = 0.5f; s->n = 0; }
        sl[i] = s;
        float p = s->p; if (p < 1e-4f) p = 1e-4f; if (p > 1 - 1e-4f) p = 1 - 1e-4f;
        st[i] = stretch(p);
    }
    st[NM] = 0.3f;
    float dot = 0; for (int i = 0; i <= NM; i++) dot += w[sel][i] * st[i];
    float pm = squash(dot); if (pm < 1e-5f) pm = 1e-5f; if (pm > 1 - 1e-5f) pm = 1 - 1e-5f;
    float sv = stretch(pm); if (sv < -7.999f) sv = -7.999f; if (sv > 7.999f) sv = 7.999f;
    float pos = (sv + 8.0f) * 2.0f; int lo = (int)pos; float fr = pos - lo;
    float *ap = apm + sel * 33;
    float pa = ap[lo] * (1 - fr) + ap[lo + 1] * fr;
    float p = 0.4f * pm + 0.6f * pa; if (p < 1e-5f) p = 1e-5f; if (p > 1 - 1e-5f) p = 1 - 1e-5f;
    { double c = -log2(bit ? p : 1 - p); total += c; tally[cur_t] += c; } ndec++;
    float err = bit - pm;
    for (int i = 0; i <= NM; i++) w[sel][i] += 0.004f * err * st[i];
    ap[lo] += 0.02f * (1 - fr) * (bit - ap[lo]); ap[lo + 1] += 0.02f * fr * (bit - ap[lo + 1]);
    for (int i = 0; i < NM; i++) { Slot *s = sl[i]; s->p += (bit - s->p) / (s->n + 1.5f); if (s->n < 255) s->n++; }
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: casecm file maskdigits [folded_out]\n"); return 2; }
    FILE *f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *s = malloc(n), *fo = malloc(n); if (fread(s, 1, n, f) != (size_t)n) return 1; fclose(f);
    int maskd = atoi(argv[2]);
    for (long i = 0; i < n; i++) {
        if (maskd && s[i] >= '0' && s[i] <= '9') s[i] = '9';
        fo[i] = isup(s[i]) ? s[i] + 32 : s[i];
    }
    if (argc > 3) { FILE *o = fopen(argv[3], "wb"); fwrite(fo, 1, n, o); fclose(o); }
    for (int i = 0; i < NM; i++) tab[i] = calloc((size_t)1 << TBITS, sizeof(Slot));
    for (int i = 0; i < 64; i++) for (int k = 0; k <= NM; k++) w[i][k] = 0.25f;
    for (int c = 0; c < 64; c++) for (int k = 0; k < 33; k++) apm[c * 33 + k] = squash((k - 16) * 0.5f);

    seen = calloc(1 << 23, 1);
    long words = 0, nonlower = 0, article = 0;
    int prevpat = 0; uint64_t prevword = 0;
    for (long a = 0; a < n;) {
        if (fo[a] == '<' && a + 5 < n && !memcmp(fo + a, "<page>", 6)) article++;
        if (!islow(fo[a])) { a++; continue; }
        long e = a; while (e < n && islow(fo[e])) e++;
        int L = (int)(e - a);
        uint64_t wh = 0x51;
        for (long i = a; i < e; i++) wh = H(wh, fo[i]);
        int nup = 0; for (long i = a; i < e; i++) nup += isup(s[i]);
        int pat;
        if (nup == 0) pat = 0;
        else if (isup(s[a]) && nup == 1) pat = 1;
        else if (nup == L) pat = 2;
        else pat = 3;
        uint64_t p2 = 0x77, p6 = 0x99, nx2 = 0x55;
        for (int k = 1; k <= 2; k++) p2 = H(p2, a - k >= 0 ? fo[a - k] : 256);
        for (int k = 1; k <= 6; k++) p6 = H(p6, a - k >= 0 ? fo[a - k] : 256);
        for (int k = 0; k < 2; k++) nx2 = H(nx2, e + k < n ? fo[e + k] : 256);
        int Lb = L < 4 ? L : 4;
        int pb = a > 0 ? fo[a - 1] : 0;
        int pbc = pb == ' ' ? 0 : pb == '[' ? 1 : pb == '\n' ? 2 : pb == '\'' ? 3 : (pb == '|' || pb == '=') ? 4 : 5;
        uint64_t ctx[NM];
        ctx[0] = H(1, wh);
        ctx[1] = H(H(2, wh), p2);
        ctx[2] = H(H(3, p6), Lb);
        ctx[3] = H(H(4, wh), prevpat);
        ctx[4] = H(H(5, prevword), wh);
        ctx[5] = H(H(H(6, nx2), Lb), p2);
        ctx[6] = H(H(H(7, prevpat), p2), Lb);
        ctx[7] = H(H(8, wh), article);
        ctx[8] = H(H(H(9, p6), nx2), prevpat);
        ctx[9] = H(H(10, wh), nx2);
        uint64_t nw = 0x33; { long q = e; while (q < n && !islow(fo[q]) && q < e + 3) q++; while (q < n && islow(fo[q])) nw = H(nw, fo[q++]); }
        ctx[10] = H(H(12, wh), nw);
        ctx[11] = H(H(H(13, p2), nw), Lb);
        ctx[12] = H(H(H(14, prevword), wh), prevpat);
        // decision tree over the pattern; node id folded into the contexts
        uint64_t c2[NM];
        int sel = pbc * 4 + prevpat;  // < 24
        cur_t = seen_word(wh) ? 0 : 4;
        code_bit(pat != 0, ctx, sel);
        if (pat != 0) {
            for (int i = 0; i < NM; i++) c2[i] = H(ctx[i], 0xB);
            cur_t += 1; code_bit(pat != 1, c2, 24 + sel % 16);
            if (pat != 1 && L > 1) {
                for (int i = 0; i < NM; i++) c2[i] = H(ctx[i], 0xC);
                cur_t += 1; code_bit(pat != 2, c2, 40 + sel % 8);
                cur_t += 1;
                if (pat == 3) {  // per-letter case bits
                    int pc = 2;
                    for (long i = a; i < e; i++) {
                        uint64_t c3[NM];
                        for (int k = 0; k < NM; k++) c3[k] = H(H(H(ctx[k], 0xD), i - a < 7 ? i - a : 7), pc);
                        c3[0] = H(H(H(11, fo[i]), pc), i - a < 3 ? i - a : 3);
                        int b = isup(s[i]);
                        code_bit(b, c3, 48 + pc * 2 + (i == a));
                        pc = b;
                    }
                }
            }
            nonlower++;
        }
        prevpat = pat; prevword = wh; words++;
        a = e;
    }
    const char *nm[8] = {"seen: lower?", "seen: Cap?", "seen: ALLCAPS?", "seen: mixed bits", "new: lower?", "new: Cap?", "new: ALLCAPS?", "new: mixed bits"};
    for (int i = 0; i < 8; i++) printf("  %-18s %9.0f bytes\n", nm[i], tally[i] / 8);
    printf("words=%ld non-lower=%ld decisions=%ld  case info = %.0f bytes (%.4f bits/word)\n",
           words, nonlower, ndec, total / 8, total / words);
    return 0;
}

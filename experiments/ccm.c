// ccm.c — generic "project out a byte class" experiment.
//
// Class C (mode u: bytes >= 0x80, mode c: UTF-8 continuation bytes 0x80..0xBF) is replaced by a
// placeholder in the skeleton (digits are always masked to '9' first, so this stacks on STC).
// Members are then coded in document order, given the whole skeleton + earlier members.
// Byte-level binarization (8 decisions, partial-byte context), context models -> mixer -> APM.
//
// usage: ccm <file> <u|c> <skeleton_out>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NM 12
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

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ccm file u|c skeleton_out\n"); return 2; }
    FILE *f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *x = malloc(n), *m = malloc(n); if (fread(x, 1, n, f) != (size_t)n) return 1; fclose(f);
    char mode = argv[2][0];
    uint8_t lo = 0x80, hi = mode == 'u' ? 0xFF : 0xBF, PH = 0x80;
    for (long i = 0; i < n; i++) {
        uint8_t c = x[i];
        m[i] = (c >= '0' && c <= '9') ? '9' : (c >= lo && c <= hi) ? PH : c;
    }
    { FILE *o = fopen(argv[3], "wb"); fwrite(m, 1, n, o); fclose(o); }
    Slot *tab[NM];
    for (int i = 0; i < NM; i++) tab[i] = calloc((size_t)1 << TBITS, sizeof(Slot));
    static float w[256][NM + 1]; for (int i = 0; i < 256; i++) for (int k = 0; k <= NM; k++) w[i][k] = 0.25f;
    float *apm = malloc(sizeof(float) * 256 * 4 * 33);
    for (int c = 0; c < 256 * 4; c++) for (int k = 0; k < 33; k++) apm[c * 33 + k] = squash((k - 16) * 0.5f);
    double total = 0; long nmem = 0, nruns = 0;
    uint8_t hist[8] = {0};  // members in document order
    uint64_t prevrun = 0;
    for (long a = 0; a < n;) {
        if (!(m[a] == PH && x[a] >= lo && x[a] <= hi)) { a++; continue; }
        long e = a; while (e < n && m[e] == PH && x[e] >= lo && x[e] <= hi) e++;
        int L = (int)(e - a), Lb = L < 8 ? L : 8;
        uint64_t l3 = 0x11, l8 = 0x22, r3 = 0x33;
        for (int k = 1; k <= 3; k++) l3 = H(l3, a - k >= 0 ? m[a - k] : 256);
        for (int k = 1; k <= 8; k++) l8 = H(l8, a - k >= 0 ? m[a - k] : 256);
        for (int k = 0; k < 3; k++) r3 = H(r3, e + k < n ? m[e + k] : 256);
        uint64_t runh = 0x44;
        for (int j = 0; j < L; j++) {
            int c = x[a + j], jj = j < 3 ? j : 3;
            int p1 = j >= 1 ? x[a + j - 1] : 256, p2 = j >= 2 ? x[a + j - 2] : 256;
            uint64_t ctx[NM];
            ctx[0] = H(H(H(1, Lb), jj), p1);
            ctx[1] = H(H(H(H(2, Lb), jj), p1), p2);
            ctx[2] = H(H(H(3, hist[0]), hist[1]), hist[2]);
            uint64_t h4 = 4; for (int k = 0; k < 4; k++) h4 = H(h4, hist[k]);
            ctx[3] = h4;
            uint64_t h6 = 6; for (int k = 0; k < 6; k++) h6 = H(h6, hist[k]);
            ctx[4] = h6;
            ctx[5] = H(H(H(7, jj), p1), l3);
            ctx[6] = H(H(H(8, jj), p1), l8);
            ctx[7] = H(H(H(9, jj), p1), r3);
            ctx[8] = H(H(H(H(10, Lb), jj), l3), r3);
            ctx[9] = H(H(H(11, jj), prevrun), p1);
            ctx[10] = H(H(H(12, jj), runh), Lb);
            ctx[11] = H(H(H(H(13, hist[0]), hist[1]), hist[2]), hist[3] + 1000 * jj);
            int partial = 1;
            for (int b = 7; b >= 0; b--) {
                int bit = (c >> b) & 1;
                float st[NM + 1]; Slot *sl[NM];
                for (int i = 0; i < NM; i++) {
                    uint64_t hh = mix64(ctx[i] ^ ((uint64_t)partial * 0xD6E8FEB86659FD93ULL));
                    Slot *s = &tab[i][hh & (((uint64_t)1 << TBITS) - 1)];
                    uint16_t chk = (uint16_t)(hh >> 48);
                    if (s->chk != chk) { s->chk = chk; s->p = 0.5f; s->n = 0; }
                    sl[i] = s;
                    float p = s->p; if (p < 1e-4f) p = 1e-4f; if (p > 1 - 1e-4f) p = 1 - 1e-4f;
                    st[i] = stretch(p);
                }
                st[NM] = 0.3f;
                int sel = partial;  // mixer weights per partial-byte node
                float dot = 0; for (int i = 0; i <= NM; i++) dot += w[sel][i] * st[i];
                float pm = squash(dot); if (pm < 1e-5f) pm = 1e-5f; if (pm > 1 - 1e-5f) pm = 1 - 1e-5f;
                float sv = stretch(pm); if (sv < -7.999f) sv = -7.999f; if (sv > 7.999f) sv = 7.999f;
                float pos = (sv + 8.0f) * 2.0f; int lo2 = (int)pos; float fr = pos - lo2;
                float *ap = apm + (partial * 4 + jj) * 33;
                float pa = ap[lo2] * (1 - fr) + ap[lo2 + 1] * fr;
                float p = 0.4f * pm + 0.6f * pa; if (p < 1e-5f) p = 1e-5f; if (p > 1 - 1e-5f) p = 1 - 1e-5f;
                total += -log2(bit ? p : 1 - p);
                float err = bit - pm;
                for (int i = 0; i <= NM; i++) w[sel][i] += 0.004f * err * st[i];
                ap[lo2] += 0.02f * (1 - fr) * (bit - ap[lo2]); ap[lo2 + 1] += 0.02f * fr * (bit - ap[lo2 + 1]);
                for (int i = 0; i < NM; i++) { Slot *s = sl[i]; s->p += (bit - s->p) / (s->n + 1.5f); if (s->n < 255) s->n++; }
                partial = partial * 2 + bit;
            }
            memmove(hist + 1, hist, 7); hist[0] = (uint8_t)c;
            runh = H(runh, c);
            nmem++;
        }
        prevrun = H(x[a], L > 1 ? x[a + 1] : 0);
        nruns++;
        a = e;
    }
    printf("mode=%c members=%ld runs=%ld  member info = %.0f bytes (%.3f bits/byte)\n", mode, nmem, nruns, total / 8, total / nmem);
    return 0;
}

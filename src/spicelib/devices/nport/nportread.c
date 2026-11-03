/**********
Enhancement-242: native n-port device -- `.nport` fit-file reader.

The `.nport` file is a compact text description of a vector-fitted Y-parameter
model, written by `pre_snp -native` and read by the native n-port device at
temperature time.  Kept as an independently testable unit (compile with
-DNPORT_TEST for a standalone reader/dumper) so the parser is validated apart
from the ngspice device framework.

Format (whitespace/newline separated, '#' to end-of-line is a comment):

    NPORT 1                 # format tag + version
    nports  N
    npoles  Np              # real-canonical layout: real poles first, then
                            #   each complex pole as an adjacent conjugate pair
    poles                   # Np lines, "re im"
      re0 im0
      ...
    d                       # N*N values, row-major Y-index (i*N+j), constant term
      ...
    e                       # N*N values, s-linear term
      ...
    res                     # N*N*Np "re im" pairs, index ((i*N+j)*Np + k)
      re im
      ...

Sections may appear in any order after the header; each is introduced by its
keyword.  All numeric fields are plain doubles.
**********/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ngspice/fteext.h"

/* token reader: skips '#'-to-EOL comments; returns 1 on a token, 0 on EOF */
static int
nport_tok(FILE *f, char *buf, int cap)
{
    int c, n = 0;
    for (;;) {
        c = fgetc(f);
        if (c == EOF) { buf[n] = '\0'; return n > 0; }
        if (c == '#') { while (c != '\n' && c != EOF) c = fgetc(f); if (n) break; else continue; }
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (n > 0) break;      /* end of a token */
            continue;              /* leading whitespace */
        }
        if (n < cap - 1) buf[n++] = (char) c;
    }
    buf[n] = '\0';
    return 1;
}

static int
nport_int(FILE *f, int *dst)
{
    char b[64];
    if (!nport_tok(f, b, sizeof b)) return 0;
    *dst = atoi(b);
    return 1;
}

static int
nport_dbl(FILE *f, double *dst)
{
    char b[64];
    if (!nport_tok(f, b, sizeof b)) return 0;
    *dst = atof(b);
    return 1;
}

/* Pure-C core: fills freshly malloc'd arrays.  Returns 0 on success, non-zero on
 * error (message in `err`).  Caller frees the arrays. */
int
snp_nport_read(const char *path,
               int *pN, int *pNp,
               double **ppoleRe, double **ppoleIm,
               double **pd, double **pe,
               double **presRe, double **presIm,
               char *err, int errlen)
{
    FILE *f = inp_pathopen(path, "r");
    int N = 0, Np = -1, ver = 0, i, got_hdr = 0;
    double *poleRe = NULL, *poleIm = NULL, *d = NULL, *e = NULL;
    double *resRe = NULL, *resIm = NULL;
    char tok[256];

    *ppoleRe = *ppoleIm = *pd = *pe = *presRe = *presIm = NULL;
    *pN = *pNp = 0;

    if (!f) { snprintf(err, errlen, "nport: cannot open fit file '%s'", path); return 1; }

#define FAIL(msg) do { snprintf(err, errlen, "nport: %s in '%s'", msg, path); goto fail; } while (0)
#define RDI(dst)  do { if (!nport_int(f, (dst))) FAIL("expected an integer"); } while (0)
#define RDD(dst)  do { if (!nport_dbl(f, (dst))) FAIL("expected a number");  } while (0)

    /* header: NPORT/nports/npoles in any order until the first section keyword */
    while (nport_tok(f, tok, sizeof tok)) {
        if      (!strcmp(tok, "NPORT"))  { RDI(&ver); got_hdr = 1; }
        else if (!strcmp(tok, "nports")) { RDI(&N); }
        else if (!strcmp(tok, "npoles")) { RDI(&Np); }
        else if (!strcmp(tok, "poles") || !strcmp(tok, "d") ||
                 !strcmp(tok, "e") || !strcmp(tok, "res")) break;
        else FAIL("unexpected header token");
    }
    if (!got_hdr || N <= 0 || Np < 0) FAIL("missing/invalid NPORT/nports/npoles header");
    if (ver != 1) FAIL("unsupported .nport format version");

    poleRe = calloc((size_t) (Np > 0 ? Np : 1), sizeof(double));
    poleIm = calloc((size_t) (Np > 0 ? Np : 1), sizeof(double));
    d      = calloc((size_t) N * N, sizeof(double));
    e      = calloc((size_t) N * N, sizeof(double));
    resRe  = calloc((size_t) N * N * (Np > 0 ? Np : 1), sizeof(double));
    resIm  = calloc((size_t) N * N * (Np > 0 ? Np : 1), sizeof(double));
    if (!poleRe || !poleIm || !d || !e || !resRe || !resIm) FAIL("out of memory");

    /* `tok` holds the first section keyword; process each section in turn */
    do {
        if      (!strcmp(tok, "poles")) { for (i = 0; i < Np; i++)     { RDD(&poleRe[i]); RDD(&poleIm[i]); } }
        else if (!strcmp(tok, "d"))     { for (i = 0; i < N * N; i++)    RDD(&d[i]); }
        else if (!strcmp(tok, "e"))     { for (i = 0; i < N * N; i++)    RDD(&e[i]); }
        else if (!strcmp(tok, "res"))   { for (i = 0; i < N*N*Np; i++)  { RDD(&resRe[i]); RDD(&resIm[i]); } }
        else FAIL("unexpected section keyword");
    } while (nport_tok(f, tok, sizeof tok));

    fclose(f);
    *pN = N; *pNp = Np;
    *ppoleRe = poleRe; *ppoleIm = poleIm;
    *pd = d; *pe = e; *presRe = resRe; *presIm = resIm;
    return 0;

fail:
    fclose(f);
    free(poleRe); free(poleIm); free(d); free(e); free(resRe); free(resIm);
    return 1;
#undef FAIL
#undef RDI
#undef RDD
}


#ifdef NPORT_TEST
int main(int argc, char **argv)
{
    int N, Np, i;
    double *pr, *pi, *d, *e, *rr, *ri;
    char err[256];
    if (argc < 2) { fprintf(stderr, "usage: %s file.nport\n", argv[0]); return 2; }
    if (snp_nport_read(argv[1], &N, &Np, &pr, &pi, &d, &e, &rr, &ri, err, sizeof err)) {
        fprintf(stderr, "%s\n", err); return 1;
    }
    printf("nports=%d npoles=%d\n", N, Np);
    for (i = 0; i < Np; i++)      printf("  pole[%d] = %g %+gj\n", i, pr[i], pi[i]);
    for (i = 0; i < N * N; i++)   printf("  d[%d]=%g e[%d]=%g\n", i, d[i], i, e[i]);
    for (i = 0; i < N*N*Np; i++)  printf("  res[%d] = %g %+gj\n", i, rr[i], ri[i]);
    free(pr); free(pi); free(d); free(e); free(rr); free(ri);
    return 0;
}
#endif

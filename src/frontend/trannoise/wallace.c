/* Wallace generator for normally distributed random variates
   Copyright: Holger Vogt, 2008
*/

//#define FASTNORM_ORIG

#ifdef HasMain
#include <sys/timeb.h>
#else
#ifndef NOSPICE
#include "ngspice/ngspice.h"
#endif
#endif
#ifdef _MSC_VER
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif
#include <math.h>
#include "ngspice/wallace.h"
#include "ngspice/FastNorm3.h"
#include "ngspice/randnumb.h"

#define POOLSIZE 4096
#define LPOOLSIZE 12
#define NOTRANS 3 /* number of (dual) transformations */

#define VE 10
#define VL (1 << VE)
#define VM (VL-1)
#define WL (4*VL)
#define WM (WL-1)

double *outgauss; /* output vector for user access */
unsigned int variate_used; /* actual index of variate called by user */
double ScaleGauss;

static double *pool1;
static double *pool2;
static unsigned int *addrif, *addrib;
static unsigned n = POOLSIZE;
static double chi1, chi2; /* chi^2 correction values */
static unsigned int newpools;


void
PolarGauss(double* py1, double* py2)
{
    double x1, x2, w;

    do {
        x1 = drand();
        x2 = drand();
        w = x1 * x1 + x2 * x2;
    } while ((w > 1.0) || (w < 0.25));

    w = sqrt((-2.0 * log(w)) / w);

    *py1 = (double)(x1 * w);
    *py2 = (double)(x2 * w);
}


void
destroy_wallace(void)
{
    tfree(pool1);
    tfree(pool2);
    tfree(addrif);
    tfree(addrib);
}


void
initw(void)
{
    unsigned i;
    double totsqr, nomsqr;
    unsigned int coa;

    /* initialize the uniform generator */
    srand((unsigned int) getpid());
    // srand(17);
    TausSeed();

    ScaleGauss = 1.;
    newpools = 1;

    /* set up the two pools */
    pool1 = TMALLOC(double, n);
    pool2 = TMALLOC(double, n);
    addrif = TMALLOC(unsigned int, (n + NOTRANS));
    addrib = TMALLOC(unsigned int, (n + NOTRANS));

    /* fill the first pool with normally distributed values */
    PolarGauss(&pool1[0], &pool1[1]);
    for (i = 1; i < n>>1; i++)
        PolarGauss(&pool1[i<<1], &pool1[(i<<1) + 1]);

    /* normalize pool content */
    /* totsqr = totsum = 0.0;
     * for (i = 0; i < n; i++) {
     *    totsqr += pool1[i] * pool1[i];
     *    totsum += pool1[i];
     * }
     * totsum = totsum/n;
     * for (i = 0; i < n; i++) {
     *    totsqr += (pool1[i] - totsum) * (pool1[i] - totsum);
     * }
     * nomsqr = sqrt(n / totsqr);
     * for (i = 0; i < n; i++)
     *    pool1[i] = (pool1[i] - totsum) * nomsqr;
     */
    totsqr = 0.0;
    for (i = 0; i < n; i++)
        totsqr += pool1[i] * pool1[i];
    nomsqr = sqrt(n / totsqr);
    for (i = 0; i < n; i++)
        pool1[i] *= nomsqr;

    /* calculate ch^2 value */
    chi1 = sqrt(sqrt(1.0 - 1.0/n));
    chi2 = sqrt(1.0 - chi1*chi1);

    /* first scaling, based on unused pool1[n-2] */
    ScaleGauss = chi1 + chi2 * ScaleGauss * pool1[n-2];
    /* access to first pool */
    outgauss = pool1;
    /* set data counter, we return n-2 values here */
    variate_used = n - 2;

    /* generate random reading addresses using a LCG */
    coa = 241;
    for (i = 0; i < (n + NOTRANS); i++) {
        // addrif[i] = s = (s * coa + cob) % (n);
        coa = CombLCGTausInt();
        addrif[i] = coa >> (32 - LPOOLSIZE);
        // printf ("Random add:\t%ld\n" , s);
    }
    coa = 193;
    for (i = 0; i < (n + NOTRANS); i++) {
        // addrib[i] = s = (s * coa + cob) % (n);
        coa = CombLCGTausInt();
        addrib[i] = coa >> (32 - LPOOLSIZE);
        // printf ("Random add:\t%ld\n" , addrib[i]);
    }

    // printf("norm for orig. Gauss: %e, chi^2 scale: %e\n", nomsqr, ScaleGauss);
    // NewWa();
}


/* original FastNorm3.c code */
#ifdef FASTNORM_ORIG
float
NewWa()
{
    int i, j, k, m;
    float p, q, r, s, t;
    int topv[6], ord[4], *top;
    float *ppt[4], *ptn;

    float nulval, endval;
    float totsqr, nomsqr;
    nulval = ScaleGauss * pool1[0];
    endval = pool1[n-1];

    /* Choose 4 random start points in the wk1[] vector
       I want them all different.  */

    top = topv + 1;
    /* Set limiting values in top[-1], top[4]  */
    top[-1] = VL;  top[4] = 0;
reran1:
    m = CombLCGTausInt();   /* positive 32-bit random */
    /* Extract two VE-sized randoms from m, which has 31 useable digits */
    m  = m >> (31 - 2*VE);
    top[0] = m & VM;  m = m >> VE;  top[1] = m & VM;
    m = CombLCGTausInt();   /* positive 32-bit random */
    /* Extract two VE-sized randoms from m, which has 31 useable digits */
    m  = m >> (31 - 2*VE);
    top[2] = m & VM;  m = m >> VE;  top[3] = m & VM;
    for (i = 0; i < 4; i++)
        ord[i] = i;
    /* Sort in decreasing size   */
    for (i = 2; i >= 0; i--)
        for (j = 0; j <= i; j++)
            if (top[j] < top[j+1]) {
                SWAP(int, top[j], top[j+1]);
                SWAP(int, ord[j], ord[j+1]);
            }

    /* Ensure all different  */
    for (i = 0; i < 3; i++)
        if (top[i] == top[i+1])
            goto reran1;

    /* Set pt pointers to their start values for the first chunk.  */
    for (i = 0; i < 4; i++) {
        j = ord[i];
        ppt[j] = pool2 + j * VL + top[i];
    }

    /* Set ptn to point into wk1  */
    ptn = pool1;

    /* Now ready to do five chunks. The length of chunk i is
       top[i-1] - top[i]  (I hope)
       At the end of chunk i, pointer ord[i] should have reached the end
       of its part, and need to be wrapped down to the start of its part.
    */
    i = 0;

chunk:
    j = top[i] - top[i-1];   /* Minus the chunk length */
    for (;  j < 0;  j++) {
        p = *ptn++;  s = *ptn++;  q = *ptn++;  r = *ptn++;
        t = (p + q + r + s) * 0.5;
        *ppt[0]++ = t - p;
        *ppt[1]++ = t - q;
        *ppt[2]++ = r - t;
        *ppt[3]++ = s - t;
    }
    /* This should end the chunk.  See if all done  */
    if (i == 4)
        goto passdone;

    /* The pointer for part ord[i] should have passed its end  */
    j = ord[i];
#ifdef dddd
    printf ("Chunk %1d done. Ptr %1d now %4d\n", i, j, ppt[j]-pool2);
#endif
    ppt[j] -= VL;
    i++;
    goto chunk;

passdone:
    /* wk1[] values have been transformed and placed in wk2[]
       Transform from wk2 to wk1 with a simple shuffle  */
    m = (CombLCGTausInt2() >> (29 - VE)) & WM;
    j = 0;
    for (i = 0; i < 4; i++)
        ppt[i] = pool1 + i * VL;
    for (i = 0; i < VL; i++) {
        p = pool2[j^m];  j++;
        s = pool2[j^m];  j++;
        q = pool2[j^m];  j++;
        r = pool2[j^m];  j++;
        t = (p + q + r + s) * 0.5;
        *ppt[0]++ = t - p;
        *ppt[1]++ = q - t;
        *ppt[2]++ = t - r;
        *ppt[3]++ = s - t;
    }

    /* renormalize again if number of pools beyond limit */
    if (!(newpools & 0xFFFF)) {
        totsqr = 0.0;
        for (i = 0; i < n; i++)
            totsqr += pool1[i] * pool1[i];
        nomsqr = sqrt(n / totsqr);
        for (i = 0; i < n; i++)
            pool1[i] *= nomsqr;
    }

    outgauss = pool1;
    /* reset data counter */
    variate_used = n - 1;

    /* set counter counting nomber of pools made */
    newpools++;

    /* new scale factor using ch^2 correction,
       using pool1[n-1] from last pool */
    ScaleGauss = chi1 + chi2 * ScaleGauss * endval;

    // printf("Pool number: %d, chi^2 scale: %e\n", newpools, ScaleGauss);

    return nulval; /* use old scale */
}

#else

/* Simplified code according to an algorithm published by C. S. Wallace:
   "Fast Pseudorandom Generators for Normal and Exponential Variates",
   ACM Transactions on Mathmatical Software, Vol. 22, No. 1, March 1996, pp. 119-127.
   Transform pool1 to pool2 and back to pool1 NOTRANS times
   by orthogonal 4 x 4 Hadamard-Matrix.
   Mixing of values is very important: Any value in the pool should contribute to
   every value in the new pools, at least after several passes (number of passes
   is set by NOTRANS to 2 or 3).
   4 values are read in a continuous sequence from the total of POOLSIZE values.
   Values are stored in steps modulo POOLSIZE/4.
   During backward transformation the values are shuffled by a random number jj.
*/

double
NewWa(void)
{
    double nulval, endval;
    double bl1, bl2, bl3, bl4; /* the four values to be transformed */
    double bsum;
    double totsqr, nomsqr;
    unsigned int i, j, jj, m, mm, mmm;

    nulval = ScaleGauss * pool1[0];
    endval = pool1[n-1];
    m = n >> 2;
    // printf("New pool after next value\n");

    /* generate new pool by transformation
       Transformation is repeated NOTRANS times */
    for (i = 0; i < NOTRANS; i++) {
        mm = m << 1;
        mmm = mm + m;
        /* forward transformation */
        // for (j = 0; j < n; j += 4) {
        for (j = 0; j < m; j++) {
            bl1 = pool1[j];
            bl2 = pool1[j+m];
            bl3 = pool1[j+mm];
            bl4 = pool1[j+mmm];
            /* Hadamard-Matrix */
            bsum = (bl1 + bl2 + bl3 + bl4) * 0.5f;
            jj = j<<2;
            pool2[jj]   = bl1 - bsum;
            pool2[jj+1] = bl2 - bsum;
            pool2[jj+2] = bsum - bl3;
            pool2[jj+3] = bsum - bl4;
        }
        /* backward transformation */
        jj = (CombLCGTausInt2() >> (31 - LPOOLSIZE)) & (n - 1);
        for (j = 0; j < m; j++) {
            bl1 = pool2[j^jj];
            bl2 = pool2[(j+m)^jj];
            bl3 = pool2[(j+mm)^jj];
            bl4 = pool2[(j+mmm)^jj];
            /* Hadamard-Matrix */
            bsum = (bl1 + bl2 + bl3 + bl4) * 0.5f;
            jj = j<<2;
            pool1[jj]   = bl1 - bsum;
            pool1[jj+1] = bl2 - bsum;
            pool1[jj+2] = bsum - bl3;
            pool1[jj+3] = bsum - bl4;
        }
    }

    /* renormalize again if number of pools beyond limit */
    if (!(newpools & 0xFFFF)) {
        totsqr = 0.0;
        for (i = 0; i < n; i++)
            totsqr += pool1[i] * pool1[i];
        nomsqr = sqrt(n / totsqr);
        for (i = 0; i < n; i++)
            pool1[i] *= nomsqr;
    }

    outgauss = pool1;
    /* reset data counter */
    variate_used = n - 1;

    /* set counter counting nomber of pools made */
    newpools++;

    /* new scale factor using ch^2 correction,
       using pool1[n-1] from previous pool */
    ScaleGauss = chi1 + chi2 * ScaleGauss * endval;

    // printf("Pool number: %d, chi^2 scale: %e\n", newpools, ScaleGauss);

    return nulval; /* use old scale */
    // return pool1[0]; /* use new scale */
}

#endif


#ifdef FASTNORMTEST
float
NewWa_not(void)
{
    float nulval, endval;
    float bl1, bl2, bl3, bl4; /* the four values to be transformed */
    float bsum;
    float totsqr, nomsqr;
    unsigned int i, j, jj;
    nulval = ScaleGauss * pool1[0];
    endval = pool1[n-1];

    // printf("New pool after next value\n");

    /* generate new pool by transformation
       Transformation is repeated NOTRANS times */
    for (i = 0; i < NOTRANS; i++) {

        /* forward transformation */
        for (j = 0; j < n; j += 4) {
            jj = j + i;
            bl1 = pool1[addrif[jj]];
            bl2 = pool1[addrif[jj+1]];
            bl3 = pool1[addrif[jj+2]];
            bl4 = pool1[addrif[jj+3]];
            /* s = (s*coa + cob) & (n - 1);
               bl1 = pool1[s];
               s = (s*coa + cob) & (n - 1);
               bl2 = pool1[s + 1];
               s = (s*coa + cob) & (n - 1);
               bl3 = pool1[s + 2];
               s = (s*coa + cob) & (n - 1);
               bl4 = pool1[s + 3];   */
            /* jj = j + i;
               bl1 = pool1[addrif[jj]];
               bl2 = pool1[addrif[jj+1]];
               bl3 = pool1[addrif[jj+2]];
               bl4 = pool1[addrif[jj+3]]; */
            /* bl1 = pool1[j];
               bl2 = pool1[j+1];
               bl3 = pool1[j+2];
               bl4 = pool1[j+3]; */
            /* Hadamard-Matrix */
            bsum = (bl1 + bl2 + bl3 + bl4) * 0.5;
            /* pool2[j] = bl1 - bsum;
               pool2[j+1] = bl2 - bsum;
               pool2[j+2] = bsum - bl3;
               pool2[j+3] = bsum - bl4; */
            pool2[addrib[jj]] = bl1 - bsum;
            pool2[addrib[jj+1]] = bl2 - bsum;
            pool2[addrib[jj+2]] = bsum - bl3;
            pool2[addrib[jj+3]] = bsum - bl4;
        }
        /* backward transformation */
        for (j = 0; j < n; j += 4) {
            bl1 = pool2[j];
            bl2 = pool2[j+1];
            bl3 = pool2[j+2];
            bl4 = pool2[j+3];
            /* bl1 = pool2[addrib[j]];
               bl2 = pool2[addrib[j+1]];
               bl3 = pool2[addrib[j+2]];
               bl4 = pool2[addrib[j+3]]; */
            /* Hadamard-Matrix */
            bsum = (bl1 + bl2 + bl3 + bl4) * 0.5;
            pool1[j] = bl1 - bsum;
            pool1[j+1] = bl2 - bsum;
            pool1[j+2] = bsum - bl3;
            pool1[j+3] = bsum - bl4;
        }
    }

    /* renormalize again if number of pools beyond limit */
    if (!(newpools & 0xFFFF)) {
        totsqr = 0.0;
        for (i = 0; i < n; i++)
            totsqr += pool1[i] * pool1[i];
        nomsqr = sqrt(n / totsqr);
        for (i = 0; i < n; i++)
            pool1[i] *= nomsqr;
    }

    outgauss = pool1;
    /* reset data counter */
    variate_used = n - 1;

    /* set counter counting nomber of pools made */
    newpools++;

    /* new scale factor using ch^2 correction,
       using pool1[n-1] from last pool */
    ScaleGauss = chi1 + chi2 * ScaleGauss * endval;

    // printf("Pool number: %d, chi^2 scale: %e\n", newpools, ScaleGauss);

    return nulval; /* use old scale */
    // return pool1[0]; /* use new scale */
}
#endif

/*      ---------------------  (test) main  -------------------------  */
/* gcc -Wall -g  -DHasMain -I../../include  wallace.c CombTaus.o -o watest.exe */
#ifdef HasMain
#include "ngspice/wallace.h"

struct timeb timenow;
struct timeb timebegin;
int sec, msec;

void
timediff(struct timeb *now, struct timeb *begin, int *sec, int *msec)
{

    *msec = now->millitm - begin->millitm;
    *sec = now->time - begin->time;
    if (*msec < 0) {
        *msec += 1000;
        (*sec)--;
    }
}


int
main()
{
    float x;
    unsigned int i;
    long int count;

    initw();
    ftime(&timebegin);
    count = 100000000;
    for (i = 0; i < count; i++) {
        x = GaussWa;
        // printf("%d\t%f\n", i, x);
    }
    ftime(&timenow);
    timediff(&timenow, &timebegin, &sec, &msec);
    printf("WallaceHV: %ld normal variates: %f s\n", count, sec + (float) msec / 1000.0);

    initnorm(0, 0);
    initnorm(77, 3);
    ftime(&timebegin);
    count = 100000000;
    for (i = 0; i < count; i++) {
        x = FastNorm;
        // printf("%d\t%f\n", i, x);
    }
    ftime(&timenow);
    timediff(&timenow, &timebegin, &sec, &msec);
    printf("FastNorm3: %ld normal variates: %f s\n", count, sec + (float) msec / 1000.0);

    return (1);
}
#endif

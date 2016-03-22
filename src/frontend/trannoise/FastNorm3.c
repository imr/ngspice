/* This is file FastNorm3.c */
/* SUPERCEDES FastNorm.c, FastNorm2.c.  Use with FastNorm3.h */
/*         24  June  2003                                    */

/* A package containing a very fast generator of pseudo-random
   Unit NORMAL variates, and some fairly high-quality UNIFORM
   generators. It also contains a straightforward implementation of
   a ChiSquared and Gamma generator copied from Ahrens and Dieter.
*/

/* Version 3 with double transformations and controllable extension
   to repeat the double transformations for higher quality at lower
   speed.
           Dated 17 May 20003.
           Copyright Christopher Stewart Wallace.
*/

/*
%A C. S. Wallace
%T Fast Pseudo-Random Generators for Normal and Exponential Variates.
%J ACM Trans. Math. Software
%V 22
%N 1
%P 119-127
%M MAR
%D 1996
%O TR 94/197, May 1994, Dept. Computer Science, Monash University
%K CSW, CSWallace, Monash, pseudo random number generator, algorithm,
   jrnl, TOMS, numbers, normal, probability, distribution, PRNG, RNG, Gaussian,
   distribution, jrnl, ACM, TOMS, TR 94 197, TR197, c1996, c199x, c19xx
*/

/*      Use of this package requires the file "FastNorm3.h" which must be
        #include-ed in any C files using this package.

        The main purpose of this package is to provide a very fast source
of pseudo-random variates from the Unit Normal N(0,1) distribution, having
the density function

        f(x) = (1/sqrt(2*PI)) * exp (-0.5 * x^2)

Variates are obtained not by calling a function, but by use of a macro
"FastNorm" defined in FastNorm3.h. In a C program, this macro may appear
anywhere a (double) expression could appear, e.g in statements like
        z += FastNorm;
        if (FastNorm < 1.1) .....
        q = fabs (FastNorm);   etc.

The revision history, and a reference to the method description, is given
later in this file under the heading "Revision history Fastnorm".

        Major sections of this file, such as the revision history and the
major subroutines, are all headed by a line containing a row of minus signs (-)
and the name of the section or subroutine.

        The generators included are:
a Uniform source of integers, unsigned integers and doubles.
Chi-sq(N)  (based on Ahrens and Dieter)
Gamma(N)        (= 0.5 * Chi-sq(2N))
Normal  (a very fast routine)
*/

/* ----------------- inclusions and some definitions ------------ */
#ifndef NOSPICE
#include "ngspice/ngspice.h"
#endif
#include <stdint.h>
#include <math.h>
#include "ngspice/FastNorm3.h"


/* --------------- (Uniform) c7rand, irandm, urandm ---------- */
/*
c  A random number generator called as a function by
c  c7rand (iseed)  or      irandm (iseed)  or urandm (iseed)

c  The parameter should be a pointer to a 2-element Sw vector.
c  The first call gives a double uniform in 0 .. 1.
c  The second gives an Sw integer uniform in 0 .. 2**31-1
c  The third gives an Sw integer with 32 bits, so unif in
c  -2**31 .. 2**31-1 if used in 32-bit signed arithmetic.
c  All update iseed[] in exactly the same way.
c  iseed[] must be a 2-element Sw vector.
c  The initial value of iseed[1] may be any 32-bit integer.
c  The initial value of iseed[0] may be any 32-bit integer except -1.
c
c  The period of the random sequence is 2**32 * (2**32-1)
c  Its quality is quite good. It is based on the mixed multiplicative
c  congruential (Lehmer) generator
           x[n+1] = (69069 * x[n] + odd constant) MOD 2^32
c  but avoids most of the well-known defects of this type of generator
c  by, in effect, generating x[n+k] from x[n] as defined by the
c  sequence above, where k is chosen randomly in 1 ... 128 with the
c  help of a subsidiary Tauseworth-type generator.
c          For the positve integer generator irandm, the less
c  significant digits are more random than is usual for a Lehmer
c  generator. The last n<31 digits do not repeat with a period of 2^n.
c  This is also true of the unsigned integer generator urandm, but less
c  so.

c  This is an implementation in C of the algorithm described in
c  Technical Report "A Long-Period Pseudo-Random Generator"
c  TR89/123, Computer Science, Monash University,
c          Clayton, Vic 3168 AUSTRALIA
c                  by
c
c          C.S.Wallace     csw@cs.monash.edu.au

c  The table mt[0:127] is defined by mt[i] = 69069 ** (128-i)
*/


#define MASK ((Sw) 0x12DD4922)
/*      or in decimal, 316492066       */
#define SCALE ((double) 1.0 / (1024.0 * 1024.0 * 1024.0 * 2.0))
/*      i.e. 2 to power -31     */


static Sw mt [128] = {
    902906369,
    2030498053,
    -473499623,
    1640834941,
    723406961,
    1993558325,
    -257162999,
    -1627724755,
    913952737,
    278845029,
    1327502073,
    -1261253155,
    981676113,
    -1785280363,
    1700077033,
    366908557,
    -1514479167,
    -682799163,
    141955545,
    -830150595,
    317871153,
    1542036469,
    -946413879,
    -1950779155,
    985397153,
    626515237,
    530871481,
    783087261,
    -1512358895,
    1031357269,
    -2007710807,
    -1652747955,
    -1867214463,
    928251525,
    1243003801,
    -2132510467,
    1874683889,
    -717013323,
    218254473,
    -1628774995,
    -2064896159,
    69678053,
    281568889,
    -2104168611,
    -165128239,
    1536495125,
    -39650967,
    546594317,
    -725987007,
    1392966981,
    1044706649,
    687331773,
    -2051306575,
    1544302965,
    -758494647,
    -1243934099,
    -75073759,
    293132965,
    -1935153095,
    118929437,
    807830417,
    -1416222507,
    -1550074071,
    -84903219,
    1355292929,
    -380482555,
    -1818444007,
    -204797315,
    170442609,
    -1636797387,
    868931593,
    -623503571,
    1711722209,
    381210981,
    -161547783,
    -272740131,
    -1450066095,
    2116588437,
    1100682473,
    358442893,
    -1529216831,
    2116152005,
    -776333095,
    1265240893,
    -482278607,
    1067190005,
    333444553,
    86502381,
    753481377,
    39000101,
    1779014585,
    219658653,
    -920253679,
    2029538901,
    1207761577,
    -1515772851,
    -236195711,
    442620293,
    423166617,
    -1763648515,
    -398436623,
    -1749358155,
    -538598519,
    -652439379,
    430550625,
    -1481396507,
    2093206905,
    -1934691747,
    -962631983,
    1454463253,
    -1877118871,
    -291917555,
    -1711673279,
    201201733,
    -474645415,
    -96764739,
    -1587365199,
    1945705589,
    1303896393,
    1744831853,
    381957665,
    2135332261,
    -55996615,
    -1190135011,
    1790562961,
    -1493191723,
    475559465,
    69069
};


double
c7rand(Sw *is)
{
    int32_t it, leh;

    it = is [0];
    leh = is [1];
    /* Do a 7-place right cyclic shift of it */
    it = ((it >> 7) & 0x01FFFFFF) + ((it & 0x7F) << 25);
    if (it >= 0)
        it = it ^ MASK;
    leh = leh * mt[it & 127] + it;
    is [0] = it;    is [1] = leh;
    if (leh < 0)
        leh = ~leh;
    return (SCALE * leh);
}


Sw
irandm(Sw *is)
{
    int32_t it, leh;

    it = is [0];
    leh = is [1];
    /* Do a 7-place right cyclic shift of it */
    it = ((it >> 7) & 0x01FFFFFF) + ((it & 0x7F) << 25);
    if (it >= 0)
        it = it ^ MASK;
    leh = leh * mt[it & 127] + it;
    is [0] = it;    is [1] = leh;
    if (leh < 0)
        leh = ~leh;
    return (leh);
}


unsigned int
urandm(Sw *is)
{
    int32_t it, leh;

    it = is [0];
    leh = is [1];
    /* Do a 7-place right cyclic shift of it */
    it = ((it >> 7) & 0x01FFFFFF) + ((it & 0x7F) << 25);
    if (it >= 0)
        it = it ^ MASK;
    leh = leh * mt[it & 127] + it;
    is [0] = it;    is [1] = leh;
    return (uint32_t) leh;
}


/* ---------------  (Chi-squared) adchi ----------------------- */
/* Simple implementation of Ahrens and Dieter method for a chi-sq
random variate of order a >> 1.  Uses c7rand, maths library */
/* 13 July 1998 */
/* Slightly faster if 'a' is the same as on previous call */
/* This routine is no longer used in the fastnorm code, but is included
because it may be useful */


static double gorder, gm, rt2gm, aold;

double
adchi(double a, int *is)
{
    double x, y, z, sq;

    if (a != aold) {
        aold = a;  gorder = 0.5 * a;
        gm = gorder - 1.0;
        rt2gm = sqrt (aold - 1.0);
    }

polar:
    x = 2.0 * c7rand(is) - 1.0;  z = c7rand(is);
    sq = x*x + z*z;
    if ((sq > 1.0) || (sq < 0.25))
        goto polar;
    y = x / z;
    x = rt2gm * y + gm;
    if (x < 0.0)
        goto polar;

    z = (1.0 + y*y) * exp (gm * log(x/gm) - rt2gm * y);
    if (c7rand(is) > z)
        goto polar;

    return (2.0 * x);
}


/* -------------------- (Gamma) rgamma (g, is) ----------- */

double
rgamma(double g, int *is)
{
    double x, y, z, sq;

    if (g != gorder) {
        gorder = g;
        gm = gorder - 1.0;      aold = 2.0 * gorder;
        rt2gm = sqrt (aold - 1.0);
    }

polar:
    x = 2.0 * c7rand(is) - 1.0;  z = c7rand(is);
    sq = x*x + z*z;
    if ((sq > 1.0) || (sq < 0.25))
        goto polar;
    y = x / z;
    x = rt2gm * y + gm;
    if (x < 0.0)
        goto polar;

    z = (1.0 + y*y) * exp (gm * log(x/gm) - rt2gm * y);
    if (c7rand(is) > z)
        goto polar;

    return (x);
}


/* ------------------  Revision history Fastnorm  ------------- */
/* Items in this revision history appear in chronological order,
so the most recent revsion appears last.
        Revision items are separated by a line of '+' characters.

        ++++++++++++++++++++++++++++++++++++++++++++++++++++++
        This is a revised version of the algorithm decribed in

        ACM Transactions on Mathematical Software, Vol 22, No 1
                March 1996, pp 119-127.

        A fast generator of pseudo-random variates from the unit Normal
distribution. It keeps a pool of about 1000 variates, and generates new
ones by picking 4 from the pool, rotating the 4-vector with these as its
components, and replacing the old variates with the components of the
rotated vector.

        The program should initialize the generator by calling initnorm(seed)
with seed a Sw integer seed value. Different seed values will give
different sequences of Normals. Seed may be any 32-bit integer.
        BUT SEE REVISION of 17 May 2003 for initnorm() parameters.
        The revised initnorm requires two integer parameters, iseed and
                quoll, the latter specifying a tradeoff between speed and
                quality.
        Then, wherever the program needs a new Normal variate, it should
use the macro FastNorm, e.g. in statements like:
        x = FastNorm;  (Sets x to a random Normal value)
or
        x += a + FastNorm * b;  (Adds a normal with mean a, SD b, to x)
        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        Changed basic formula, which was:
                t = (p+q+r+s)*0.5; p = p-t; q = t-q; r = t-r; s = t-s;
        This gives sum of new p+q+r+s = 2p(old) which may not be a great
choice. The new version is:
                t = (p+q+r+s)*0.5; p = p-t; q = q-t; r = t-r; s = t-s;
        which gives new p+q+r+s = p+q-r-s (old) which may be better.
        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        Revision 14 November 1998
        The older version "FastNorm" which was available via ftp was found
to have a defect which could affect some applications.

        Dr Christine Rueb, (Max Planck Institut fur Infomatik,
                Im Stadtwald W 66123 Saabrucken, F.G.R.,
                        (rueb@mpi-sb.mpg.de)

found that if a large number N of consecutive variates were summed to give
a variate S with nominally N(0,N) distribution, the variance of S was in some
cases too small. The effect was noticed with N=400, and was particularly strong
for N=1023 if the first several (about 128) variates from FastNorm were
discarded. Dr. Rueb traced the effect to an unexpected tendency of FastNorm
to concentrate values with an anomolous correlation into the first 128
elements of the variate pool.
        With the help of her analysis, the algorithm has been revised in a
way which appears to overcome the problem, at the cost of about a 19%
reduction in speed (which still leaves the method very fast.)

        IT  MUST  BE  RECOGNISED  THAT  THIS  ALGORITHM  IS  NOVEL
AND  WHILE  IT  PASSES  A  NUMBER  OF  STANDARD  TESTS  FOR  DISTRIBUTIONAL
FORM,  LACK  OF  SERIAL  CORRELATION  ETC.,  IT  MAY  STILL  HAVE  DEFECTS.

RECALL  THE  NUMBER  OF  YEARS  WHICH  IT  TOOK  FOR  THE  LIMITATIONS  OF
THE  LEHMER  GENERATOR  FOR  UNIFORM  VARIATES  TO  BECOME  APPARENT !!!

UNTIL  MORE  EXPERIENCE  IS  GAINED  WITH  THIS  TYPE  OF  GENERATOR,  IT
WOULD  BE  WISE  IN  ANY  CRITICAL  APPLICATION  TO  COMPARE  RESULTS
OBTAINED  USING  IT  WITH  RESULTS  OBTAINED  USING  A  "STANDARD"  FORM
OF  GENERATOR  OF  NORMAL  VARIATES  COUPLED  WITH  A  WELL-DOCUMENTED
GENERATOR  OF  UNIFORM  VARIATES.
        ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        Revision 1 April 2003.
        Trying a scanning process proposed by R.P.Brent. It needs 2 pool
vectors, as it cannot update in-situ, but may be more robust.
        It is a bit slower on a 133 Mhz PC but just as fast on a newer PC
(moggie) at about 16 ns per call in the 'speed.c' test.
        The extreme-value defect is the same on old and new versions.
If one finds a value 'A' such that a batch of B genuine Normal variates has
probability 0.2 of containing a variate with abolute value greater than A,
then the probability that both of two consecive batches of B will contain
such a value should be 0.2 times 0.2, or 0.04. Instead, both versions give
the extreme value prob. as 0.200 (over a million batches) but give the
consective-pair prob as 0.050 for batch size B = 1024.
        ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        Revision 17 May 2003.
        The fundamental defect of the method, namely an inadequate 'mixing'
of squared value ('energy') between one generation of the pool and the next,
cannot readily be removed. In going from one pool to the next, the energy
in an old variate is shared among just 4 variates in the new pool. Hence it
takes many generations before the energy of some original variate can be
distributed across the whole pool. The number of generations needed cannot
be less than the log to base 4 of the pool size, or 5 for a pool size of
1024. In fact, the pseudo-random indexing of the pool means that rather
more generations are needed on average.
        The defect is readily revealed by the following test. One picks a
"batch size" comparable to the pool size, say 500 or 1000. One then
computes a value A such that a batch will with probability 0.2 contain one
or more variates with absolute value exceeding A.
One then draws batches from FastNorm,
and tests each batch to see if it contains such an extreme value.
Over many batches, one counts the frequency of such 'extreme' batches,
and finds (with FastNorm2) that it is indeed about 0.2. However, when one counts
the frequency with which succesive batches are both extreme, one finds it to
be higher than the proper value (0.2)^2 = 0.04. For batch sizes round the pool
size, it can be as high as 0.05. That is, although the frequncy of extreme
values is about right, their occurrence in the stream is correlated over a
scale of the order of the pool size.
        The same correlation effect is seen in the average 4th moment of
successive batches.
        Since this inter-generational correlation cannot be avoided, the
this revision seeks to reduce it by performing at least two simple
rotations of the pool at each generation. Obviously, some speed is lost,
but the correlations are reduced.
        To allow the user to trade off speed and quality, the initialization
function initnorm() now provides a QUALITY parameter 'quoll' which controls
how many double-rotations are done for each generation.
        See the comments in initnorm() for more detail.
        ++++++++++  End of revision notes  +++++++++ */



/* -----------------  Some test results  ------------------------ */
/*
General form:
        Some simple tests were conducted by transforming FastNorm variates
in several ways to yield a variable nominally uniformly distributed in 0 ... 1.
Uniformity of the derived variate was then tested by a ChiSquared test on a
100-cell histogram with cell counts around 10000. These tests are crude, but
showed no untoward results on the present version.
        Transformations included:
        y = 0.5 * (1.0 + erf (n1 / sqrt(2))

        y = 0.5 * (n1 / (n1^2 + n2^2 + n3^2) - 1)

        y = exp (-0.5 * (n1^2 + n2^2))

        y = (n1^2 + n2^2) / (n1^2 + n2^2 + n3^2 + n4^2)

                where n1, n2 etc are successive Normal variates.
It may be noted that some of these are sensitive to serial correlation if
present.

Fourth moment of batches:
        Extensive tests for correlation among the fourth moments of successive
batches of variates were made, with batch sizes comparabe to or (worst case)
equal to the size of the variate pool (4096 in this revision).
        With 'quality' 1, significant correlation appears after 10^6 batches
of worst-case size.
        With quality 2, no significant correlation is evident after 10^7
batches. A just-significant correlation appears after 3.6*10^7 batches.
As this requires some 1.4*10^11 deviates to be drawn, it may be irrelevent
for many applications. The observed correlation coefficent was 0.0008.
        With quality 3, results are OK after 10^8 batches, or more than
4*10^11 variates.
        No tests have been done with quality 4 as yet.

Speed:
        Speed tests were done on a PC running RedHat Linux, using "-O"
compiler optimization. The test loop was
        for (i = 0; i < 500000000; i++) {
            a += FastNorm;  a -= FastNorm;
        }
        Thus the test makes 10^9 uses of FastNorm. The time taken, (which
includes time for a call in 'initnorm' and the loop overhead) depends on
the 'quality' set by initnorm.
        Quality 1:      21.5 sec
        Quality 2:      32.1 sec
        Quality 3:      42.5 sec
        Quality 4:      53.1 sec

By way of comparison, the same 10^9 call loop was timed with the Unix library
"random()" routine substituted for FastNorm, and the variable 'a' defined as
integer rather than double.  Also, since most use of a Uniform generator such
as "random()" requires that the returned integer be scaled into a floating-
point number in 0 ... 1, the timing was repeated with
        "a += random" ('a' integer) replaced by "a += Scale*random()" where
'a' is double and Scale = 2^(-31). The times obtained were:
        Random (integer): 44.1 sec
        Random (double) : 47.7 sec

        It can be seen that FastNorm (even at quality 3) is faster than a
commonly-used Uniform generator. To some extent, this result may vary on
different computers and compilers. Since FastNorm (at least for qualities
above 1) no doubt does more arithmetic per variate than "random()", much of
its speed advantage must come from its replacement of a function call per
variate by a macro which makes only one function call every 4095 variates.
Computers with lower 'call' overheads than the PC used here might show
differnt results.
        Incidently, the Uniform generator 'c7rand()' included in this
package, which returns a double uniform in 0 ... 1, and is of fairly high
quality, gives a time in the same test of 36.8 sec, a little faster than
'random()'.
*/


/* -----------------  globals  ------------------------- */
/* A pool must have a length which is a multiple of 4.
 * During regeneration of a new pool, the pool is treated as 4
 * consecutive vectors, each of length VL.
 */

#define VE 10
#define VL (1 << VE)
#define VM (VL-1)
#define WL (4*VL)
#define WM (WL-1)

Sw gaussfaze;
Sf *gausssave;
Sf GScale;
/* GScale,fastnorm,gaussfaze, -save must be visible to callers */
static Sf chic1, chic2; /* Constants used in getting ChiSq_WL */
Sw gslew;               /*  Counts generations */
static Sw qual;         /*  Sets number of double transforms per generation. */
static Sw c7g [2];      /*  seed values for c7rand */

Sf wk1 [WL], wk2 [WL];          /*  Pools of variates. */


/* ------------------  regen  ---------------------- */
/* Takes variates from wk1[], transforms to wk[2], then back to wk1[]. */

static void
regen(void)
{
    Sw i, j, m;
    Sf p, q, r, s, t;
    Sw topv[6], ord[4], *top;
    Sf *ppt[4], *ptn;

    /* Choose 4 random start points in the wk1[] vector
       I want them all different. */

    top = topv + 1;
    /* Set limiting values in top[-1], top[4] */
    top[-1] = VL;  top[4] = 0;
reran1:
    m = irandm (c7g);   /* positive 32-bit random */
    /* Extract two VE-sized randoms from m, which has 31 useable digits */
    m  = m >> (31 - 2*VE);
    top[0] = m & VM;  m = m >> VE;  top[1] = m & VM;
    m = irandm (c7g);   /* positive 32-bit random */
    /* Extract two VE-sized randoms from m, which has 31 useable digits */
    m  = m >> (31 - 2*VE);
    top[2] = m & VM;  m = m >> VE;  top[3] = m & VM;
    for (i = 0; i < 4; i++)
        ord[i] = i;
    /* Sort in decreasing size */
    for (i = 2; i >= 0; i--)
        for (j = 0; j <= i; j++)
            if (top[j] < top[j+1]) {
                SWAP(Sw, top[j], top[j+1]);
                SWAP(Sw, ord[j], ord[j+1]);
            }

    /* Ensure all different */
    for (i = 0; i < 3; i++)
        if (top[i] == top[i+1])
            goto reran1;

    /* Set pt pointers to their start values for the first chunk. */
    for (i = 0; i < 4; i++) {
        j = ord[i];
        ppt[j] = wk2 + j * VL + top[i];
    }

    /* Set ptn to point into wk1 */
    ptn = wk1;

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
    /* This should end the chunk.  See if all done */
    if (i == 4)
        goto passdone;

    /* The pointer for part ord[i] should have passed its end */
    j = ord[i];
#ifdef dddd
    printf ("Chunk %1d done. Ptr %1d now %4d\n", i, j, ppt[j]-wk2);
#endif
    ppt[j] -= VL;
    i++;
    goto chunk;

passdone:
    /* wk1[] values have been transformed and placed in wk2[]
       Transform from wk2 to wk1 with a simple shuffle */
    m = (irandm (c7g) >> (29 - VE)) & WM;
    j = 0;
    for (i = 0; i < 4; i++)
        ppt[i] = wk1 + i * VL;
    for (i = 0; i < VL; i++) {
        p = wk2[j^m];  j++;
        s = wk2[j^m];  j++;
        q = wk2[j^m];  j++;
        r = wk2[j^m];  j++;
        t = (p + q + r + s) * 0.5;
        *ppt[0]++ = t - p;
        *ppt[1]++ = q - t;
        *ppt[2]++ = t - r;
        *ppt[3]++ = s - t;
    }

    /* We have a new lot of variates in wk1 */
}


/* -------------------  renormalize  --------------------------- */
/* Rescales wk1[] so sum of squares = WL */
/* Returns the original sum-of-squares */

Sf
renormalize(void)
{
    Sf ts, vv;
    Sw i;

    ts = 0.0;
    for (i = 0; i < WL; i++)
        ts += wk1[i] * wk1[i];

    vv = sqrt (WL / ts);
    for (i = 0; i < WL; i++)
        wk1[i] *= vv;
    return (ts);
}


/* ------------------------  BoxMuller  ---------------------- */
/* Fills block gvec of length ll with proper normals */

static void
boxmuller(Sf *gvec, Sw ll)
{
    Sw i;
    Sf tx, ty, tr, tz;

    /* Here, replace the whole pool with conventional Normal variates */
    i = 0;
nextpair:
    tx = 2.0 * c7rand(c7g) - 1.0;  /* Uniform in -1..1 */
    ty = 2.0 * c7rand(c7g) - 1.0;  /* Uniform in -1..1 */
    tr = tx * tx + ty * ty;
    if ((tr > 1.0) || (tr < 0.25))
        goto nextpair;
    tz = -2.0 * log (c7rand(c7g));  /* Sum of squares */
    tz = sqrt(tz / tr);
    gvec [i++] = tx * tz;   gvec [i++] = ty * tz;
    if (i < ll)
        goto nextpair;
    /* Horrid, but good enough */
}


/* -------------------------  initnorm  ---------------------- */
/* To initialize, given a seed integer and a quality level.
   The seed can be any integer. The quality level quoll should be
   between 1 and 4. Quoll = 1 gives high speed, but leaves some
   correlation between the 4th moments of successive batches of values.
   Higher values of quoll give lower speed but less correlation.

   If called with quoll = 0, initnorm performs a check that the
   most crucial routine (regen) is performing correctly. In this
   case, the value of 'iseed' is ignored. Initnorm will report the
   results of the test, which compares pool values with check17 and
   check98, which are defined below.
   When a check call is made, a proper call on initnorm must then
   be made before using the FastNorm macro. A check call does not
   properly initialize the routines even if it succeeds.
*/

static Sf check17 = 0.1255789;
static Sf check98 = -0.7113293;

void
initnorm(Sw seed, Sw quoll)
{
    Sw i;

    /* At one stage, we need to generate a random variable Z such that
       (WL * Z*Z) has a Chi-squared-WL density. Now, a var with
       an approximate Chi-sq-K distn can be got as
       (A + B*n)**2 where n has unit Normal distn,
       A**2 = K * sqrt (1 - 1/K),  A**2 + B**2 = K.  (For large K)
       So we form Z as (1/sqrt(WL)) * (A + B*n)
       or   chic1 + chic2 * n   where
       chic1 = A / sqrt(WL), chic2 = B / sqrt(WL).
       Hence
       chic1 = sqrt(A*A / WL) = sqrt(sqrt(1 - 1/WL)),
       chic2 = sqrt(1 - chic1*chic1)
    */

    chic1 = sqrt(sqrt(1.0 - 1.0 / WL));
    chic2 = sqrt(1.0 - chic1 * chic1);

    /* Set regen counter "gslew" which will affect renormalizations.
       Since pools are OK already, we wont't need to renorm for a
       while */
    gslew = 1;
    /* Finally, set "gaussfaze" to return all of wk1
     * except the last entry at WL-1 */
    gaussfaze = WL-1;
    gausssave = wk1;

    /* If quoll = 0, do a check on installation */
    if (quoll == 0)
        goto docheck;
    qual = quoll;
    /* Check sensible values for quoll, say 1 to 4 */
    if ((quoll < 0) || (quoll > 4)) {
        printf ("From initnorm(): quoll parameter %d out of\
 range 1 to 4\n", quoll);
        return;
    }
    c7g[0] = seed;  c7g[1] = -3337792;

    /* Fill wk1[] with good normals */
    boxmuller (wk1,  WL);
    /* Scale so sum-of-squares = WL */
    GScale = sqrt (renormalize () / WL);
    /* We have set
       GScale to restore the original ChiSq_WL sum-of-squares */
    return;

docheck:
    /* Set a simple pattern in wk1[] and test results of regen */
    for (i = 0; i < WL; i++)
        wk1[i] = wk2[i] = 0.0;
    wk1[0] = sqrt ((double) WL);
    c7g[0] = 1234567;  c7g[1] = 9876543;
    for (i = 0; i < 60; i++)
        regen();
    /* Check a couple of values */
    if ((fabs (wk1[17] - check17) > 0.00001) ||
        (fabs (wk1[98] - check98) > 0.00001))
    {
        printf ("\nInitnorm check failed.\n");
        printf ("Expected %8.5f got %10.7f\n", check17, wk1[17]);
        printf ("Expected %8.5f got %10.7f\n", check98, wk1[98]);
    } else {
        printf ("\nInitnorm check OK\n");
    }
}


/* ----------------------  fastnorm  -------------------------- */
/* If gslew shows time is ripe, renormalizes the pool
              fastnorm() returns the value GScale*gausssave[0]. */

Sf
fastnorm(void)
{
    Sf sos;
    Sw n1;

    if (!(gslew & 0xFFFF))
        sos = renormalize ();

    /* The last entry of gausssave, at WL-1, will not have been used.
       Use it to get an approx. to sqrt (ChiSq_WL / WL).
       See initnorm() code for details */
    GScale = chic1 + chic2 * GScale * gausssave [WL-1];
    for (n1 = 0; n1 < qual; n1++)
        regen ();
    gslew++;

    gaussfaze = WL - 1;

    return (GScale * gausssave [0]);
}


/* ---------------------  (test) main  ------------------------- */

#ifdef Main
#include "ngspice/FastNorm3.h"

int
main()
{
    Sf x;  Sw i;
    initnorm (0, 0);
    initnorm (77, 2);
    printf ("SoS %20.6f\n", renormalize());
    // for (i = 0; i < 2000000; i++)
    //    x = FastNorm;
    for (i = 0; i < 200; i++) {
        x = FastNorm;
        printf("%d\t%f\n", i, x);
    }
    printf ("SoS %20.6f\n", renormalize());
    exit (1);
}

#endif

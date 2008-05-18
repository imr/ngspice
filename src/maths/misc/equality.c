/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
**********/

#include <assert.h>
#include "ngspice.h"

#ifdef _MSC_VER
typedef __int64 long64;
#else
typedef long long long64;
#endif

#define Abs(x) ((x) < 0 ? -(x) : (x))

/* From Bruce Dawson, Comparing floating point numbers,
   http://www.cygnus-software.com/papers/comparingfloats/Comparing%20floating%20point%20numbers.htm
   Original this function is named AlmostEqual2sComplement but we leave it to AlmostEqualUlps
   and can leave the code (measure.c, dctran.c) unchanged. The transformation to the 2's complement
   prevent problems around 0.0. 
   One Ulp is equivalent to a maxRelativeError of between 1/4,000,000,000,000,000 and 1/8,000,000,000,000,000.
   Practical: 3 < maxUlps < some hundred's (or thousand's) - depending on numerical requirements. 
*/
bool AlmostEqualUlps(double A, double B, int maxUlps)
{
    long64 aInt, bInt, intDiff;

    if (A == B)
        return TRUE;

    /* If not - the entire method can not work */
    assert(sizeof(double) == sizeof(long64)); 

    /* Make sure maxUlps is non-negative and small enough that the */
    /* default NAN won't compare as equal to anything. */
    assert(maxUlps > 0 && maxUlps < 4 * 1024 * 1024);
    aInt = *(long64*)&A;
    /* Make aInt lexicographically ordered as a twos-complement int */
    if (aInt < 0)
#ifdef _MSC_VER
        aInt = 0x8000000000000000 - aInt;
#else
        aInt = 0x8000000000000000LL - aInt;
#endif
    bInt = *(long64*)&B;
    /* Make bInt lexicographically ordered as a twos-complement int */
    if (bInt < 0)
#ifdef _MSC_VER
        bInt = 0x8000000000000000 - bInt;
#else
        bInt = 0x8000000000000000LL - bInt;
#endif
#ifdef _MSC_VER
    intDiff = Abs(aInt - bInt);
#else
    intDiff = llabs(aInt - bInt);
#endif
/*    printf("A:%e B:%e aInt:%d bInt:%d  diff:%d\n", A, B, aInt, bInt, intDiff); */
    if (intDiff <= maxUlps)
        return TRUE;
    return FALSE;
}

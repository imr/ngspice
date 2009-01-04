#include "ngspice.h"

/* this is really ugly - but it is a emergency case */

#ifndef HAVE_DECL_ISINF
#ifndef HAVE_ISINF

static int
isinf(double x)
{
    volatile double a = x;

    if (a > DBL_MAX)
        return 1;
    if (a < -DBL_MAX)
        return -1;
    return 0;
}

/*
 * end isinf.c
 */
#else /* HAVE_ISINF */
int Dummy_Symbol_5;
#endif /* HAVE_ISINF */
#endif /* HAVE_DECL_ISINF */

/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * Standard utility routines.
 * Most moved to MISC/
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpstd.h"


/* This might not be around.  If not then forget about sorting. */

#ifndef HAVE_QSORT
#ifndef qsort
int compar(const void* a, const void* b);

int compar(const void* a, const void* b) {
    NG_IGNORE(a);
    NG_IGNORE(b);
    return 0;
}

void qsort(void* base, size_t num, size_t size,
    int (*compar)(const void* a, const void* b))
{
    NG_IGNORE(base);
    NG_IGNORE(num);
    NG_IGNORE(size);
    NG_IGNORE(compar);
}

#endif
#endif

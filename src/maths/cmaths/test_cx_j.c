#include <stdio.h>

#include <config.h>
#include <memory.h>
#include <dvec.h>
#include <complex.h>

#include "cmath.h"
#include "cmath1.h"

FILE *cp_err;

int
main(void)
{
    complex *c = NULL;
    complex *d = NULL;
    short int t1;
    short int t2;
    int n1;
    int n2;

    cp_err = stderr;
    n1 = 1;
    t1 = VF_COMPLEX;
    c = alloc_c(n1);
    realpart(&c[0]) = .0;
    imagpart(&c[0]) = 1.0;
    d = (complex *) cx_j((void *) c, t1, n1, &n2, &t2);
    if (realpart(&d[0]) == -1 && imagpart(&d[0]) == 0)
	return 0;
    else
	return 1;
}

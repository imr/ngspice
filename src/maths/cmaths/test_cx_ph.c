#include <stdio.h>
#include <math.h>
#include <float.h>

#include "ngspice/config.h"
#include "ngspice/memory.h"
#include "ngspice/dvec.h"
#include "ngspice/complex.h"

#include "ngspice/defines.h"
#include "cmath.h"
#include "cmath1.h"

FILE *cp_err;

int
main(void)
{
    ngcomplex_t *c = NULL;
    double *d = NULL;
    short int t1;
    short int t2;
    int n1;
    int n2;
    double eps = DBL_EPSILON;

    cp_err = stderr;
    n1 = 1;
    t1 = VF_COMPLEX;
    c = alloc_c(n1);
    realpart(c[0]) = .0;
    imagpart(c[0]) = 1.0;
    d = (double *) cx_ph((void *) c, t1, n1, &n2, &t2);
    if (M_PI/2 - eps < d[0] && d[0] < M_PI/2 + eps)
	return 0;
    else
	return 1;
}

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

static int eq_p(double a, double b, double eps)
{
    return fabs(a-b) < eps;
}


int
main(void)
{
    ngcomplex_t *c = NULL;
    double *d = NULL;
    short int t1;
    short int t2;
    int n1, n2;
    double eps = DBL_EPSILON;

    cp_err = stderr;
    n1 = 9;
    t1 = VF_COMPLEX;
    c = alloc_c(n1);

    realpart(c[0]) =  0.0; imagpart(c[0]) = +1.0;  /* i^1 */
    realpart(c[1]) = -1.0; imagpart(c[1]) =  0.0;  /* i^2 */
    realpart(c[2]) =  0.0; imagpart(c[2]) = -1.0;  /* i^3 */
    realpart(c[3]) = +1.0; imagpart(c[3]) =  0.0;  /* i^4 */
    realpart(c[4]) =  0.0; imagpart(c[4]) = +1.0;  /* i^5 */
    realpart(c[5]) = +1.0; imagpart(c[5]) =  0.0;  /* i^4 */
    realpart(c[6]) =  0.0; imagpart(c[6]) = -1.0;  /* i^3 */
    realpart(c[7]) = -1.0; imagpart(c[7]) =  0.0;  /* i^2 */
    realpart(c[8]) =  0.0; imagpart(c[8]) = +1.0;  /* i^1 */

    d = (double *) cx_cph((void *) c, t1, n1, &n2, &t2);

    if ( eq_p(1*M_PI/2, d[0], eps) &&
         eq_p(2*M_PI/2, d[1], eps) &&
         eq_p(3*M_PI/2, d[2], eps) &&
         eq_p(4*M_PI/2, d[3], eps) &&
         eq_p(5*M_PI/2, d[4], eps) &&
         eq_p(4*M_PI/2, d[5], eps) &&
         eq_p(3*M_PI/2, d[6], eps) &&
         eq_p(2*M_PI/2, d[7], eps) &&
         eq_p(1*M_PI/2, d[8], eps) )
	return 0;
    else
	return 1;
}

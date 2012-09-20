/* Copyright: Holger Vogt, 2008
 Discrete simulation of colored noise and stochastic
 processes and 1/fa power law noise generation
 Kasdin, N.J.;
 Proceedings of the IEEE
 Volume 83,  Issue 5,  May 1995 Page(s):802 - 827
*/

#include <math.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdarg.h>                     // var. argumente
#include "ngspice/1-f-code.h"
#include "ngspice/ngspice.h"

#include "ngspice/fftext.h"
#include "ngspice/wallace.h"


void
f_alpha(int n_pts, int n_exp, double X[], double Q_d, double alpha)
{
    unsigned int i;
    double *hfa, *wfa;
    double ha;

    ha = alpha/2.0f;
    // Q_d = sqrt(Q_d); /* find the deviation of the noise */
    hfa = TMALLOC(double,n_pts);
    wfa = TMALLOC(double,n_pts);
    hfa[0] = 1.0f;
    wfa[0] = Q_d * GaussWa;
    /* generate the coefficients hk */
    for (i = 1; i < n_pts; i++) {
        /* generate the coefficients hk */
        hfa[i] = hfa[i-1] * (ha + (double)(i-1)) /  (double)(i));
        /* fill the sequence wk with white noise */
        wfa[i] = Q_d * GaussWa;
    }

    // for (i = 0; i < n_pts; i++)
    //    printf("rnd %e, hk %e\n", wfa[i], hfa[i]);

    /* perform the discrete Fourier transform */
    fftInit(n_exp);
    rffts(hfa, n_exp, 1);
    rffts(wfa, n_exp, 1);

    /* multiply the two complex vectors */
    rspectprod(hfa, wfa, X, n_pts);
    /* inverse transform */
    riffts(X, n_exp, 1);

    free(hfa);
    free(wfa);
    /* fft tables will be freed in vsrcaccept.c and isrcaccept.c
       fftFree(); */
    fprintf(stdout, "%d (2e%d) one over f values created\n", n_pts, n_exp);
}

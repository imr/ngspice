/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Gary W. Ng
**********/

/*
 * Nintegrate.c (noizDens, lnNdens, lnNlstDens, data)
 *
 *    This subroutine evaluates the integral of the function
 *
 *                                             EXPONENT
 *                      NOISE = a * (FREQUENCY)
 *
 *   given two points from the curve.  If EXPONENT is relatively close
 *   to 0, the noise is simply multiplied by the change in frequency.
 *   If it isn't, a more complicated expression must be used.  Note that
 *   EXPONENT = -1 gives a different equation than EXPONENT <> -1.
 *   Hence, the reason for the constant 'N_INTUSELOG'.
 */

#include "ngspice/ngspice.h"
#include "ngspice/noisedef.h"

#define limexp(x) (x > 700 ? exp(700.0)*(1.0+x-700.0) : exp(x))

double
Nintegrate (double noizDens, double lnNdens, double lnNlstDens, Ndata *data)
{
    double exponent;
    double a;

    exponent = (lnNdens - lnNlstDens) / data->delLnFreq;
    if ( fabs(exponent) < N_INTFTHRESH ) {
	return (noizDens * data->delFreq);
    } else {
	a = limexp(lnNdens - exponent*data->lnFreq);
	exponent += 1.0;
	if (fabs(exponent) < N_INTUSELOG) {
	    return (a * (data->lnFreq - data->lnLastFreq));
        } else {
	    return (a * ((exp(exponent * data->lnFreq) - exp(exponent * data->lnLastFreq)) /
		    exponent));
        }
    }
}

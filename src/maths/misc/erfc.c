/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numconst.h"

#ifndef HAVE_ERFC

/* erfc computes the erfc(x) the code is from sedan's derfc.f */
double erfc (double x)
{
    double sqrtPi, n, temp1, xSq, sum1, sum2;
    sqrtPi = sqrt( PI );
    x = ABS( x );
    n = 1.0;
    xSq = 2.0 * x * x;
    sum1 = 0.0;

    if ( x > 3.23 ) {
	/* asymptotic expansion */
	temp1 = exp( - x * x ) / ( sqrtPi * x );
	sum2 = temp1;

	while ( sum1 != sum2 ) {
	    sum1 = sum2;
	    temp1 = -1.0 * ( temp1 / xSq );
	    sum2 += temp1;
	    n += 2.0;
	}
	return( sum2 );
    }
    else {
	/* series expansion for small x */
	temp1 = ( 2.0 / sqrtPi ) * exp( - x * x ) * x;
	sum2 = temp1;
	while ( sum1 != sum2 ) {
	    n += 2.0;
	    sum1 = sum2;
	    temp1 *= xSq / n;
	    sum2 += temp1;
	}
        return( 1.0 - sum2 );
    }
}

/* From C. Hastings, Jr., Approximations for digital computers,
	Princeton Univ. Press, 1955.
   Approximation accurate to within 1.5E-7
   (making some assumptions about your machine's floating point mechanism)
*/
double
ierfc(double  x)     
{
  double t, z;
  
  t =  1/(1 + 0.3275911*x);
  z =  1.061405429;
  z = -1.453152027 + t * z;
  z =  1.421413741 + t * z;
  z = -0.284496736 + t * z;
  z =  0.254829592 + t * z;
  z =  exp(-x*x) * t * z;
  
  return(z);
}

#else
int Dummy_Symbol_5;
#endif

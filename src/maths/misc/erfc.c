/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice.h"
#include "numglobs.h"
#include "numconst.h"

/* erfc computes the erfc(x) the code is from sedan's derfc.f */

double erfc ( double x)
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


/* Paolo Nenzi 2002 - This program tests function
 * implementations.
 */


/*
 * The situation on erfc functions in spice/cider:
 *
 * First we have the ierfc in spice, a sort of interpolation, which is
 * fast to compute but gives is not so "good"
 * Then we have derfc from cider, which is accurate but slow
 * then we have glibc implementation.
 *
 * Proposal: 
 *
 * Use Linux implementation as defualt and then test cider one.
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fpu_control.h>




double 
derfc ( double x)
{
    double sqrtPi, n, temp1, xSq, sum1, sum2;
    sqrtPi = sqrt( M_PI );
    x = fabs( x );
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



int main (void)
{
 fpu_control_t prec;
 double x = -100.0;
 double y1= 0.0, y2 = 0.0;
 
// _FPU_GETCW(prec);
// prec &= ~_FPU_EXTENDED;
// prec |= _FPU_DOUBLE;
// _FPU_SETCW(prec);


for (;(x <= 100.0);)
{
    y1 = ierfc(x);
    y2 = derfc(x);
    printf("A: %e \t s: %e \t c: %e \t (c-lin): %e\n", x, y1, y2, y2-erfc(x) );
     x = x + 0.1;
    } 
 exit(1);
}

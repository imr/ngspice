/* Paolo Nenzi 2002 - This program tests erfc function
 * implementations.
 */


/*
 * The situation on erfc functions in spice/cider:
 *
 * First we have the ierfc in spice, a sort of interpolation, which is
 * fast to compute but gives is not so "good"
 * Then we have derfc from cider, which is accurate but slow, the code is from sedan's derfc.f .
 * Both above are only valid for x > 0.0
 * Then we have glibc/os specific implementation.
 *
 * Proposal: 
 *
 * Use glibc/os specific implementation as default and then test cider one.
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifdef HAVE_FPU_CTRL
#include <fpu_control.h>
#endif


double 
derfc(double x)
{
    double sqrtPi, n, temp1, xSq, sum1, sum2;
    sqrtPi = sqrt( M_PI );
    x = fabs( x ); /* only x > 0 interested */
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
 double x = -30.0;
 double y1= 0.0, y2 = 0.0;
 
#ifdef HAVE_FPU_CTRL
 fpu_control_t prec;
 _FPU_GETCW(prec);
 prec &= ~_FPU_EXTENDED;
 prec |= _FPU_DOUBLE;
 _FPU_SETCW(prec);
#endif

for (;(x <= 30.0);)
   {
     y1 = ierfc(x);
     y2 = derfc(x);
     printf("x: %f \t ierfc: %e \t derfc: %e \t erfc: %e\n", x, y1, y2, erfc(x) );
     x = x + 1.0;
   }
 exit(1);
}

/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
**********/

/*
 * Missing math functions
 */
#include <config.h>
#include "ngspice.h"
#include <stdio.h>
#include "missing_math.h"



#ifndef HAVE_LOGB

double
logb(double x)	
{
  double y = 0.0;
  
  if (x != 0.0) 
    {
      if (x < 0.0)
	x = - x;
      while (x > 2.0) 
	{
	  y += 1.0;
	  x /= 2.0;
	}
      while (x < 1.0) 
	{
	  y -= 1.0;
	  x *= 2.0;
	}
    } 
  else
    y = 0.0;
  
  return y;
}
#endif




#ifndef HAVE_SCALB
#  ifdef HAVE_SCALBN
#    define scalb scalbn
#else                       /* Chris Inbody */

double
scalb(double x, int n)
{
  double y, z = 1.0, k = 2.0;
  
  if (n < 0) {
    n = -n;
    k = 0.5;
  }
  
  if (x != 0.0)
    for (y = 1.0; n; n >>= 1) {
      y *= k;
      if (n & 1)
	z *= y;
    }
  
  return x * z;
}
#   endif /* HAVE_SCALBN */
#endif /* HAVE_SCALB */


#ifndef HAVE_ERFC
/* From C. Hastings, Jr., Approximations for digital computers,
	Princeton Univ. Press, 1955.
   Approximation accurate to within 1.5E-7
   (making some assumptions about your machine's floating point mechanism)
*/

double
erfc(double  x)
     
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
#endif

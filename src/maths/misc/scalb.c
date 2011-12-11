/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
**********/

#include "ngspice/ngspice.h"

#ifndef HAVE_SCALBN

double
scalbn(double x, int n)
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

#else /* HAVE_SCALBN */

int Dummy_Symbol_1;

#endif /* HAVE_SCALBN */


#ifndef HAVE_SCALB

double
scalb(double x, double n)
{
  return scalbn(x, (int) n);
}

#else /* HAVE_SCALB */

int Dummy_Symbol_2;

#endif /* HAVE_SCALB */

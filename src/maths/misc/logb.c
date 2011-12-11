/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
**********/

#include "ngspice/ngspice.h"

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
#else
int Dummy_Symbol_3;
#endif

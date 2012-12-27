/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Wayne A. Christopher, U. C. Berkeley CAD Group 
**********/

/*
 * All the functions used in the parse tree.  These functions return HUGE
 * if their argument is out of range.
 */

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/fteext.h"
#include "ngspice/ifsim.h"
#include "ngspice/inpptree.h"
#include "inpxx.h"

/* XXX These should be in math.h */


#ifndef HAVE_ATANH
extern double asinh(), acosh(), atanh();
#endif 

double PTfudge_factor;

#define	MODULUS(NUM,LIMIT)	((NUM) - ((int) ((NUM) / (LIMIT))) * (LIMIT))

double
PTabs(double arg)
{
    return arg >= 0.0 ? arg : -arg;
}

double
PTsgn(double arg)
{
    return arg > 0.0 ? 1.0 : arg < 0.0 ? -1.0 : 0.0;
}

double
PTplus(double arg1, double arg2)
{
    return (arg1 + arg2);
}

double
PTminus(double arg1, double arg2)
{
    return (arg1 - arg2);
}

double
PTtimes(double arg1, double arg2)
{
    return (arg1 * arg2);
}

double 
PTdivide(double arg1, double arg2)
{
    if (arg2 >= 0.0)
	arg2 += PTfudge_factor;
    else
	arg2 -= PTfudge_factor;

    if (arg2 == 0.0)
        return (HUGE);

    return (arg1 / arg2);
}

double
PTpower(double arg1, double arg2)
{
    if (arg1 < 0.0) {
	if (fabs(arg2 - ((int) arg2)) / (arg2 + 0.001) < 0.000001) {
	    arg2 = (int) arg2;
	} else {
	    arg1 = -arg1;
	}
    }
    return (pow(arg1, arg2));
}

double
PTmin(double arg1, double arg2)
{
    return arg1 > arg2 ? arg2 : arg1;
}

double
PTmax(double arg1, double arg2)
{
    return arg1 > arg2 ? arg1 : arg2;
}

double
PTacos(double arg)
{
    return (acos(arg));
}

double
PTacosh(double arg)
{
#ifdef HAVE_ACOSH
    return (acosh(arg));
#else
    if (arg < 1.0)
	arg = 1.0;
    return (log(arg + sqrt(arg*arg-1.0)));
#endif 
}

double
PTasin(double arg)
{
    return (asin(arg));
}

double
PTasinh(double arg)
{
#ifdef HAVE_ASINH
    return (asinh(arg));
#else
    return log(arg + sqrt(arg * arg + 1.0));
#endif 
}

double
PTatan(double arg)
{
    return (atan(arg));
}

double
PTatanh(double arg)
{
#ifdef HAVE_ATANH
    return (atanh(arg));
#else
    if (arg < -1.0)
	arg = -1.0 + PTfudge_factor + 1e-10;
    else if (arg > 1.0)
	arg = 1.0 - PTfudge_factor - 1e-10;
    return (log((1.0 + arg) / (1.0 - arg)) / 2.0);
#endif 
}

double
PTustep(double arg)
{
    if (arg < 0.0)
	return 0.0;
    else if (arg > 0.0)
	return 1.0;
    else
	return 0.5; /* Ick! */
}

/* MW. PTcif is like "C" if - 0 for (arg<=0), 1 elsewhere */

double
PTustep2(double arg)
{
    if (arg <= 0.0) 
	return 0.0;
    else if (arg <= 1.0)
	return arg;
    else /* if (arg > 1.0) */
	return 1.0;
}

double
PTeq0(double arg)
{
    return (arg == 0.0) ? 1.0 : 0.0;
}

double
PTne0(double arg)
{
    return (arg != 0.0) ? 1.0 : 0.0;
}

double
PTgt0(double arg)
{
    return (arg > 0.0) ? 1.0 : 0.0;
}

double
PTlt0(double arg)
{
    return (arg < 0.0) ? 1.0 : 0.0;
}

double
PTge0(double arg)
{
    return (arg >= 0.0) ? 1.0 : 0.0;
}

double
PTle0(double arg)
{
    return (arg <= 0.0) ? 1.0 : 0.0;
}

double
PTuramp(double arg)
{
    if (arg < 0.0)
	return 0.0;
    else
	return arg;
}

double
PTcos(double arg)
{
    return (cos(MODULUS(arg, 2 * M_PI)));
}

double
PTcosh(double arg)
{
    return (cosh(arg));
}

double
PTexp(double arg)
{
    return (exp(arg));
}

double
PTln(double arg)
{
    if (arg < 0.0)
        return (HUGE);
    return (log(arg));
}

double
PTlog(double arg)
{
    if (arg < 0.0)
        return (HUGE);
    return (log10(arg));
}

double
PTsin(double arg)
{
    return (sin(MODULUS(arg, 2 * M_PI)));
}

double
PTsinh(double arg)
{
    return (sinh(arg));
}

double
PTsqrt(double arg)
{
    if (arg < 0.0)
        return (HUGE);
    return (sqrt(arg));
}

double
PTtan(double arg)
{
    return (tan(MODULUS(arg, M_PI)));
}

double
PTtanh(double arg)
{
    return (tanh(arg));
}

double
PTuminus(double arg)
{
    return (- arg);
}

double
PTpwl(double arg, void *data)
{
  struct pwldata { int n; double *vals; } *thing = (struct pwldata *) data;

  double y;

  int k0 = 0;
  int k1 = thing->n/2 - 1;

  while(k1-k0 > 1) {
    int k = (k0+k1)/2;
    if(thing->vals[2*k] > arg)
      k1 = k;
    else
      k0 = k;
  }

  y = thing->vals[2*k0+1] +
    (thing->vals[2*k1+1] - thing->vals[2*k0+1]) *
    (arg - thing->vals[2*k0]) / (thing->vals[2*k1] - thing->vals[2*k0]);

  return y;
}

double
PTpwl_derivative(double arg, void *data)
{
  struct pwldata { int n; double *vals; } *thing = (struct pwldata *) data;

  double y;

  int k0 = 0;
  int k1 = thing->n/2 - 1;

  while(k1-k0 > 1) {
    int k = (k0+k1)/2;
    if(thing->vals[2*k] > arg)
      k1 = k;
    else
      k0 = k;
  }

  y =
    (thing->vals[2*k1+1] - thing->vals[2*k0+1]) /
    (thing->vals[2*k1]   - thing->vals[2*k0]);

  return y;
}

double
PTceil(double arg1)
{
    return (ceil(arg1));
}

double
PTfloor(double arg1)
{
    return (floor(arg1));
}


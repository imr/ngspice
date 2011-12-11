#include "ngspice/ngspice.h"

#if !HAVE_DECL_ISNAN
#ifndef HAVE_ISNAN

/* isnan (originally) for SOI devices in MINGW32  hvogt (dev.c) */
union ieee754_double
{
  double d;
  
  /* This is the IEEE 754 double-precision format.  */
  struct
  {
    /* Together these comprise the mantissa.  */
    unsigned int mantissa1:32;
    unsigned int mantissa0:20;
    unsigned int exponent:11;
    unsigned int negative:1;
  } ieee;
  struct
  {
    /* Together these comprise the mantissa.  */
    unsigned int mantissa1:32;
    unsigned int mantissa0:19;
    unsigned int quiet_nan:1;
    unsigned int exponent:11;
    unsigned int negative:1;
  } ieee_nan;
};

int
isnan(double value)
{
  union ieee754_double u;

  u.d = value;

  /* IEEE 754 NaN's have the maximum possible
                exponent and a nonzero mantissa.  */
  return ((u.ieee.exponent & 0x7ff) == 0x7ff &&
          (u.ieee.mantissa0 != 0 || u.ieee.mantissa1 != 0));

}

/*
 * end isnan.c
 */
#else /* HAVE_ISNAN */
int Dummy_Symbol_4;
#endif /* HAVE_ISNAN */
#endif /* HAVE_DECL_ISNAN */

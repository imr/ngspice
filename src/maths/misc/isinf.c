#include "ngspice.h"

#if (HAVE_DECL_ISINF < 1)

#ifndef HAVE_ISINF

#ifdef HAVE_IEEEFP_H

  int isinf(double x) { return !finite(x) && x==x; }

#else /* HAVE_IEEEFP_H */

  /* this is really ugly - but it is a emergency case */
  
  static int
  isinf(double x)
  {
      volatile double a = x;
  
      if (a > DBL_MAX)
          return 1;
      if (a < -DBL_MAX)
          return -1;
      return 0;
  }

#endif /* HAVE_IEEEFP_H */

#else /* HAVE_ISINF */

  int Dummy_Symbol_5;

#endif /* HAVE_ISINF */

#endif /* HAVE_DECL_ISINF */

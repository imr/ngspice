#include "ngspice/ngspice.h"

#if !HAVE_DECL_ISINF
#ifndef HAVE_ISINF
#if defined(HAVE_FINITE)

/* not the best replacement - see missing_math.h */

int isinf(double x) { return !finite(x) && x==x; }

#else /* HAVE_FINITE */

/* this is really ugly - but it is a emergency case */

static int
isinf (const double x)
{
  double y = x - x;
  int s = (y != y);

  if (s && x > 0)
    return +1;
  else if (s && x < 0)
    return -1;
  else
    return 0;
}

#endif /* HAVE_FINITE */
#else /* HAVE_ISINF */
int Dummy_Symbol_7;
#endif /* HAVE_ISINF */
#endif /* HAVE_DECL_ISINF */

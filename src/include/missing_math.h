/**********
Copyright 1999 Emmanuel Rouat
**********/


/* Decl. for missing maths functions, if any */

#ifndef MISSING_MATH_H_INCLUDED
#define MISSING_MATH_H_INCLUDED

bool AlmostEqualUlps(double, double, int);

#ifndef HAVE_ERFC
extern double erfc(double);
#endif

#ifndef HAVE_LOGB
extern double logb(double);
#endif

#ifndef HAVE_SCALB
extern double scalb(double, double);
#endif

#ifndef HAVE_SCALBN
extern double scalbn(double, int);
#endif

#ifndef HAVE_DECL_ISNAN
#ifndef HAVE_ISNAN
extern int isnan(double);
#endif
#endif

#ifndef HAVE_DECL_ISINF
# ifndef HAVE_ISINF
#  if defined(HAVE_FINITE) && (defined (HAVE_DECL_ISNAN) || defined (HAVE_ISNAN))
#   define isinf(x) (!finite(x) && !isnan(x))
#  endif
# else
extern int isinf(double);
# endif
#endif

#endif /* MISSING_MATH_H_INCLUDED */

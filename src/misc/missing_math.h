/*************
 * Header file for missing_math.c
 * 1999 E. Rouat
 ************/

#ifndef MISSING_MATH_H_INCLUDED
#define MISSING_MATH_H_INCLUDED

bool AlmostEqualUlps(float, float, int);

#ifndef HAVE_ERFC
double erfc(double);
#endif

#ifndef HAVE_LOGB
double logb(double);
#endif

#ifndef HAVE_SCALB
#  ifndef HAVE_SCALBN
double scalb(double, int);
#  endif
#endif

#ifndef HAVE_ISNAN
int isnan(double);
#endif

#endif /* MISSING_MATH_H_INCLUDED */

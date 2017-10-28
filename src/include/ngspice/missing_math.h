/**********
Copyright 1999 Emmanuel Rouat
**********/

/* Decl. for missing maths functions, if any */

#ifndef ngspice_MISSING_MATH_H
#define ngspice_MISSING_MATH_H

bool AlmostEqualUlps(double, double, int);

#ifndef HAVE_LOGB
extern double logb(double);
#endif

#ifndef HAVE_SCALB
extern double scalb(double, double);
#endif

#ifndef HAVE_SCALBN
extern double scalbn(double, int);
#endif

#if !HAVE_DECL_ISNAN
#ifndef HAVE_ISNAN
extern int isnan(double);
#endif
#endif

#if !HAVE_DECL_ISINF
#ifndef HAVE_ISINF
#if defined(HAVE_FINITE) && (HAVE_DECL_ISNAN || defined (HAVE_ISNAN))
#define isinf(x) (!finite(x) && !isnan(x))
#define HAVE_ISINF
#else
#ifdef HAVE_IEEEFP_H
extern int isinf(double);
#endif
#endif
#else  /* HAVE_ISINF */
extern int isinf(double);
#endif /* HAVE_ISINF */
#endif /* HAVE_DECL_ISINF */

#endif

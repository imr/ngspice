/** \file cmath.h
    \brief Header file for cmath*.c
*/

#ifndef ngspice_CMATH_H
#define ngspice_CMATH_H

#define alloc_c(len)    (TMALLOC(ngcomplex_t, len))
#define alloc_d(len)    (TMALLOC(double, len))

#endif

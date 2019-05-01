/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */
#ifndef ngspice_CONST_H
#define ngspice_CONST_H

#define CONSTsqrt2 1.4142135623730950488016887242097
#define CONSTpi 3.1415926535897932384626433832795
#define CONSTnap 2.7182818284590452353602874713527
#define CONSTlog10e 0.43429448190325182765112891891661
#define CONSTlog2e 1.4426950408889634073599246810019


/* https://physics.nist.gov/cgi-bin/cuu/Value?c
 * value = 299 792 458 m s-1 (exact) */
#define CONSTc 299792458

/* https://www.nist.gov/pml/weights-and-measures/si-units-temperature
 * Note that for general use in an expression, the negative value must
 * be guarded by (), so CONSTKtoC_for_str should only be used for building
 * a string value */
#define CONSTCtoK 273.15
#define CONSTKtoC_for_str -CONSTCtoK
#define CONSTKtoC (CONSTKtoC_for_str)

/* https://physics.nist.gov/cgi-bin/cuu/Value?e
 *                value = 1.602 176 6208 x 10-19 C
 * standard uncertainty = 0.000 000 0098 x 10-19 C */
#define CHARGE 1.6021766208e-19

/* https://physics.nist.gov/cgi-bin/cuu/Value?k
 *                value = 1.380 648 52 x 10-23 J K-1
 * standard uncertainty = 0.000 000 79 x 10-23 J K-1 */
#define CONSTboltz 1.38064852e-23

/* https://physics.nist.gov/cgi-bin/cuu/Value?h
 *                value = 6.626 070 040 x 10-34 J s
 * standard uncertainty = 0.000 000 081 x 10-34 J s */
#define CONSTplanck 6.626070040e-34

#define CONSTmuZero (4.0 * CONSTpi * 1E-7) /* MuZero H/m */

/* epsilon zero  e0*u0*c*c=1 */
#define CONSTepsZero (1.0 / (CONSTmuZero * CONSTc * CONSTc)) /* F/m */

/* This value is not really constant over temperature and frequency, but
 * 3.9 is the most common "all-purpose" value */
#define CONSTepsrSiO2 3.9

#define CONSTepsSiO2 (CONSTepsrSiO2 * CONSTepsZero)  /* epsilon SiO2 F/m */
#define REFTEMP (27.0 + CONSTCtoK) /* 27 degrees C in K */


/* Some global variables defining constant values */
extern double CONSTroot2;
extern double CONSTvt0;
extern double CONSTKoverQ;
extern double CONSTe;

#endif /* include guard */

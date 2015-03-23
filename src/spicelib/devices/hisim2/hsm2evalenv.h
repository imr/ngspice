/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2014 Hiroshima University & STARC

 MODEL NAME : HiSIM
 ( VERSION : 2  SUBVERSION : 8  REVISION : 0 )
 
 FILE : hsm2evalenv.h

 Date : 2014.6.5

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

#ifndef HSM2_EVAL_ENV_H
#define HSM2_EVAL_ENV_H

/* macros and constants used in hsm2eval2yz.c */

/*---------------------------------------------------*
* Numerical constants. (macro) 
*-----------------*/

/* machine epsilon */
#if defined(_FLOAT_H) && defined(DBL_EPSILON)
#define C_EPS_M (DBL_EPSILON) 
#else
#define C_EPS_M (2.2204460492503131e-16) 
#endif

#define MAX_EXP 5.834617425e14
#define MIN_EXP 1.713908431e-15
#define EXP_THR 34.0

/* sqrt(2) */
#define C_SQRT_2 (1.414213562373095e+00) 

/* 1/3 */
#define C_1o3   (3.333333333333333e-01) 
/* 2/3 */
#define C_2o3   (6.666666666666667e-01) 
/* 2^(1/3) */
#define C_2p_1o3    (1.259921049894873e+00) 

/* Pi */
#define C_Pi   (3.141592653589793)
#define C_Pio2 (1.570796326794897)

/* Unit change */
#define C_m2cm    (1.0e2)
#define C_m2cm_p2 (1.0e4)
#define C_m2cm_p1o2 (1.0e1)
#define C_m2um    (1.0e6)
#define C_cm2m_p3 (1.0e-6)
#define C_vmax    (1.0e5)
#define C_E0_p2   (1.0e9)

/*---------------------------------------------------*
* Physical constants/properties. (macro) 
*-----------------*/
/* Elemental charge */
#define C_QE    (1.6021918e-19) 

/* Boltzmann constant */
#define C_KB    (1.3806226e-23) 

/* Permitivity of Si, SiO2 and vacuum */
#define C_ESI   (1.034943e-10) 
#define C_EOX   (3.453133e-11) 
#define C_VAC   (8.8541878e-12) 

/* Room temperature constants */
#define C_T300  (300e+00)
#define C_b300  (3.868283e+01)
/* #define C_Eg0   (1.1785e0) */ /*changed to parameter sIN.eg0*/

/* Build-in potential */
/*#define C_Vbi   (1.0e0)*/ /* changed to parameter sIN.vbi */


/* Intrinsic carrier density at 300K */
#define C_Nin0  (1.04e+16)


/*---------------------------------------------------*
* Functions. (macro)  Take care of the arguments.
*-----------------*/
#define Fn_Sqr(x)   ( (x)*(x) ) /* x^2 */
#define Fn_Max(x,y) ( (x) >= (y) ? (x) : (y) ) /* max[x,y] */
#define Fn_Min(x,y) ( (x) <= (y) ? (x) : (y) ) /* min[x,y] */
#define Fn_Sgn(x)   ( (x) >= 0  ?  (1) : (-1) )    /* sign[x] */

#endif /* HSM2_EVAL_ENV_H */

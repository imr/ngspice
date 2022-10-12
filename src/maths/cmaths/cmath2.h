/*************
 * 1999 E. Rouat
 * 3-Clause BSD
 ************/

 /** \file cmath2.h
     \brief Header file for cmath2.c, function prototypes
 */

#ifndef ngspice_CMATH2_H
#define ngspice_CMATH2_H



void * cx_norm(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_uminus(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_rnd(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_sunif(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_sgauss(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_poisson(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_exponential(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_mean(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_stddev(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_length(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_vector(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_unitvec(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_plus(void *data1, void *data2, short int datatype1, short int datatype2, int length);
void * cx_minus(void *data1, void *data2, short int datatype1, short int datatype2, int length);
void * cx_times(void *data1, void *data2, short int datatype1, short int datatype2, int length);
void * cx_mod(void *data1, void *data2, short int datatype1, short int datatype2, int length);
void * cx_max(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_min(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_d(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_avg(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_floor(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_ceil(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_nint(void *data, short int type, int length, int *newlength, short int *newtype);


#endif




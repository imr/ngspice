/*************
 * 1999 E. Rouat 
 * 3-Clause BSD
 ************/

 /** \file cmath1.h
     \brief Header file for cmath1.c, function prototypes
 */

#ifndef ngspice_CMATH1_H
#define ngspice_CMATH1_H


void * cx_mag(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_ph(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_cph(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_unwrap(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_j(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_real(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_imag(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_conj(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_pos(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_db(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_log10(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_log(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_exp(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_sqrt(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_sin(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_sinh(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_cos(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_cosh(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_tan(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_tanh(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_atan(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_atanh(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_sortorder(void *data, short int type, int length, int *newlength, short int *newtype);


#endif

/*************
 * Header file for cmath1.c
 * 1999 E. Rouat
 ************/

#ifndef CMATH1_H_INCLUDED
#define CMATH1_H_INCLUDED


void * cx_mag(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_ph(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_j(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_real(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_imag(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_pos(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_db(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_log(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_ln(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_exp(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_sqrt(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_sin(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_cos(void *data, short int type, int length, int *newlength, short int *newtype);



#endif

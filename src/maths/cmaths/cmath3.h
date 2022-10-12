/*************
 * 1999 E. Rouat
 * 3-Clause BSD
 ************/

 /** \file cmath3.h
     \brief Header file for cmath3.c, function prototypes
 */

#ifndef ngspice_CMATH3_H
#define ngspice_CMATH3_H


void * cx_divide(void *data1, void *data2, short int datatype1, short int datatype2, int length);
void * cx_comma(void *data1, void *data2, short int datatype1, short int datatype2, int length);
void * cx_power(void *data1, void *data2, short int datatype1, short int datatype2, int length);
void * cx_eq(void *data1, void *data2, short int datatype1, short int datatype2, int length);
void * cx_gt(void *data1, void *data2, short int datatype1, short int datatype2, int length);
void * cx_lt(void *data1, void *data2, short int datatype1, short int datatype2, int length);
void * cx_ge(void *data1, void *data2, short int datatype1, short int datatype2, int length);
void * cx_le(void *data1, void *data2, short int datatype1, short int datatype2, int length);
void * cx_ne(void *data1, void *data2, short int datatype1, short int datatype2, int length);


#endif

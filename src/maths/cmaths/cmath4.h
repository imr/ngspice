/*************
 * Header file for cmath4.c
 * 1999 E. Rouat
 ************/

#ifndef CMATH4_H_INCLUDED
#define CMATH4_H_INCLUDED

void * cx_and(void *data1, void *data2, short int datatype1, short int datatype2, int length);
void * cx_or(void *data1, void *data2, short int datatype1, short int datatype2, int length);
void * cx_not(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_interpolate(void *data, short int type, int length, int *newlength, 
		      short int *newtype, struct plot *pl, struct plot *newpl, int grouping);
void * cx_deriv(void *data, short int type, int length, int *newlength, short int *newtype, 
		struct plot *pl, struct plot *newpl, int grouping);


#endif

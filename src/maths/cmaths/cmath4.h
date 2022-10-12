/*************
 * 1999 E. Rouat
 * 3-Clause BSD
 ************/

 /** \file cmath4.h
	 \brief Header file for cmath4.c, function prototypes
 */

#ifndef ngspice_CMATH4_H
#define ngspice_CMATH4_H

void * cx_and(void *data1, void *data2, short int datatype1, short int datatype2, int length);
void * cx_or(void *data1, void *data2, short int datatype1, short int datatype2, int length);
void * cx_not(void *data, short int type, int length, int *newlength, short int *newtype);
void * cx_interpolate(void *data, short int type, int length, int *newlength, 
		      short int *newtype, struct plot *pl, struct plot *newpl, int grouping);
void * cx_deriv(void *data, short int type, int length, int *newlength, short int *newtype, 
		struct plot *pl, struct plot *newpl, int grouping);
void * cx_integ(void *data, short int type, int length, int *newlength, short int *newtype,
		struct plot *pl, struct plot *newpl, int grouping);
void * cx_group_delay(void *data, short int type, int length, int *newlength, short int *newtype, 
		struct plot *pl, struct plot *newpl, int grouping);
void * cx_fft(void *data, short int type, int length, int *newlength, short int *newtype,
              struct plot *pl, struct plot *newpl, int grouping);
void * cx_ifft(void *data, short int type, int length, int *newlength, short int *newtype,
               struct plot *pl, struct plot *newpl, int grouping);

#endif

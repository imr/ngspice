/*
 * Copyright (c) 1985 Thomas L. Quarles
 * Modified: Paolo Nenzi 1999, Arno W. Peters 2000
 */
#ifndef _COMPLEX_H
#define _COMPLEX_H


/* Complex numbers. */
struct _complex {   /* IBM portability... */
    double cx_real;
    double cx_imag;
} ;

typedef struct _complex complex;

#define realpart(cval)  ((struct _complex *) (cval))->cx_real
#define imagpart(cval)  ((struct _complex *) (cval))->cx_imag


/*  header file containing definitions for complex functions
 *
 *  Each expects two arguments for each complex number - a real and an
 *  imaginary part.
 */
typedef struct {
    double real;
    double imag;
} SPcomplex;


#define DC_ABS(a,b) (fabs(a) + fabs(b))


/*
 * Division among complex numbers
 */
#define DC_DIVEQ(a,b,c,d) { \
    double r,s,x,y;\
    if(fabs(c)>fabs(d)) { \
        r=(d)/(c);\
        s=(c)+r*(d);\
        x=((*(a))+(*(b))*r)/s;\
        y=((*(b))-(*(a))*r)/s;\
    } else { \
        r=(c)/(d);\
        s=(d)+r*(c);\
        x=((*(a))*r+(*(b)))/s;\
        y=((*(b))*r-(*(a)))/s;\
    }\
    (*(a)) = x; \
    (*(b)) = y; \
}

/*
 * This is the standard multiplication among complex numbers:
 * (x+jy)=(a+jb)*(c+jd)
 * x = ac - bd and y = ad + bc
 */
#define DC_MULT(a,b,c,d,x,y) { \
    *(x) = (a) * (c) - (b) * (d) ;\
    *(y) = (a) * (d) + (b) * (c) ;\
}


/*
 * Difference among complex numbers a+jb and c+jd
 * a = a - c   amd b = b - d
 */
#define DC_MINUSEQ(a,b,c,d) { \
    *(a) -= (c) ;\
    *(b) -= (d) ;\
}

/*
 * Square root among complex numbers 
 * We need to treat all the cases because the sqrt() function
 * works only on real numbers.
 */
#define	C_SQRT(A) {							      \
	double	_mag, _a;						      \
	if ((A).imag == 0.0) {						      \
	    if ((A).real < 0.0) {					      \
		(A).imag = sqrt(-(A).real);				      \
		(A).real = 0.0;						      \
	    } else {							      \
		(A).real = sqrt((A).real);				      \
		(A).imag = 0.0;						      \
	    }								      \
	} else {							      \
	    _mag = sqrt((A).real * (A).real + (A).imag * (A).imag);	      \
	    _a = (_mag - (A).real) / 2.0;				      \
	    if (_a <= 0.0) {						      \
		(A).real = sqrt(_mag);					      \
		(A).imag /= (2.0 * (A).real); /*XXX*/			      \
	    } else {							      \
		_a = sqrt(_a);						      \
		(A).real = (A).imag / (2.0 * _a);			      \
		(A).imag = _a;						      \
	    }								      \
	}								      \
    }

/*
 * This macro calculates the squared modulus of the complex number
 * and return it as the real part of the same number:
 * a+jb -> a = (a*a) + (b*b)
 */ 
#define	C_MAG2(A) (((A).real = (A).real * (A).real + (A).imag * (A).imag),    \
	(A).imag = 0.0)

/*
 * Two macros to obtain the colpex conjugate of a number,
 * The first one replace the given complex with the conjugate,
 * the second sets A as the conjugate of B.
 */
#define	C_CONJ(A) ((A).imag *= -1.0)

#define	C_CONJEQ(A,B) {							      \
	(A).real = (B.real);						      \
	(A).imag = - (B.imag);						      \
    }

/*
 * Simple assignement
 */
#define	C_EQ(A,B) {							      \
	(A).real = (B.real);						      \
	(A).imag = (B.imag);						      \
    }

/*
 * Normalization ???
 * 
 */


#define	C_NORM(A,B) {							      \
	if ((A).real == 0.0 && (A).imag == 0.0) {			      \
	    (B) = 0;							      \
	} else {							      \
	    while (fabs((A).real) > 1.0 || fabs((A).imag) > 1.0) {	      \
		(B) += 1;						      \
		(A).real /= 2.0;					      \
		(A).imag /= 2.0;					      \
	    }								      \
	    while (fabs((A).real) <= 0.5 && fabs((A).imag) <= 0.5) {	      \
		(B) -= 1;						      \
		(A).real *= 2.0;					      \
		(A).imag *= 2.0;					      \
	    }								      \
	}								      \
    }

/*
 * The magnitude of the complex number 
 */
   
#define	C_ABS(A) (sqrt((A).real * (A.real) + (A.imag * A.imag)))

/* 
 * Standard arithmetic between complex numbers
 * 
 */

#define	C_MUL(A,B) {							      \
	double	TMP1, TMP2;						      \
	TMP1 = (A.real);						      \
	TMP2 = (B.real);						      \
	(A).real = TMP1 * TMP2 - (A.imag) * (B.imag);			      \
	(A).imag = TMP1 * (B.imag) + (A.imag) * TMP2;			      \
    }

#define	C_MULEQ(A,B,C) {						      \
	(A).real = (B.real) * (C.real) - (B.imag) * (C.imag);		      \
	(A).imag = (B.real) * (C.imag) + (B.imag) * (C.real);		      \
    }

#define	C_DIV(A,B) {							      \
	double	_tmp, _mag;						      \
	_tmp = (A.real);						      \
	(A).real = _tmp * (B.real) + (A).imag * (B.imag);		      \
	(A).imag = - _tmp * (B.imag) + (A.imag) * (B.real);		      \
	_mag = (B.real) * (B.real) + (B.imag) * (B.imag);		      \
	(A).real /= _mag;						      \
	(A).imag /= _mag;						      \
    }

#define	C_DIVEQ(A,B,C) {						      \
	double	_mag;							      \
	(A).real = (B.real) * (C.real) + (B.imag) * (C.imag);		      \
	(A).imag = (B.imag) * (C.real) - (B.real) * (C.imag) ;		      \
	_mag = (C.real) * (C.real) + (C.imag) * (C.imag);		      \
	(A).real /= _mag;						      \
	(A).imag /= _mag;						      \
    }

#define	C_ADD(A,B) {							      \
	(A).real += (B.real);						      \
	(A).imag += (B.imag);						      \
    }

#define	C_ADDEQ(A,B,C) {						      \
	(A).real = (B.real) + (C.real);					      \
	(A).imag = (B.imag) + (C.imag);					      \
    }

#define	C_SUB(A,B) {							      \
	(A).real -= (B.real);						      \
	(A).imag -= (B.imag);						      \
    }

#define	C_SUBEQ(A,B,C) {						      \
	(A).real = (B.real) - (C.real);					      \
	(A).imag = (B.imag) - (C.imag);					      \
    }



#endif /*CMPLX*/

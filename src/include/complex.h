/*
 * Copyright (c) 1985 Thomas L. Quarles
 * Modified: 1999 Paolo Nenzi, 2000 Arno W. Peters
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


/*
 *  Each expects two arguments for each complex number - a real and an
 *  imaginary part.
 */
typedef struct {
    double real;
    double imag;
} SPcomplex;

/*
 *  COMPLEX NUMBER DATA STRUCTURE
 *
 *  >>> Structure fields:
 *  Real  (RealNumber)
 *      The real portion of the number.  Real must be the first
 *      field in this structure.
 *  Imag  (RealNumber)
 *      The imaginary portion of the number. This field must follow
 *      immediately after Real.
 */

#define spREAL  double

/* Begin `RealNumber'. */
typedef  spREAL  RealNumber, *RealVector;

/* Begin `ComplexNumber'. */
typedef  struct
{   RealNumber  Real;
    RealNumber  Imag;
} ComplexNumber, *ComplexVector;


/* Some defines used mainly in cmath.c. */
#define alloc_c(len)    ((complex *) tmalloc((len) * sizeof (complex)))
#define alloc_d(len)    ((double *) tmalloc((len) * sizeof (double)))
#define FTEcabs(d)  (((d) < 0.0) ? - (d) : (d))
#define cph(c)    (atan2(imagpart(c), (realpart(c))))
#define cmag(c)  (sqrt(imagpart(c) * imagpart(c) + realpart(c) * realpart(c)))
#define radtodeg(c) (cx_degrees ? ((c) / 3.14159265358979323846 * 180) : (c))
#define degtorad(c) (cx_degrees ? ((c) * 3.14159265358979323846 / 180) : (c))
#define rcheck(cond, name)      if (!(cond)) { \
    fprintf(cp_err, "Error: argument out of range for %s\n", name); \
    return (NULL); }


#define cdiv(r1, i1, r2, i2, r3, i3)            \
{                           \
    double r, s;                    \
    if (FTEcabs(r2) > FTEcabs(i2)) {          \
        r = (i2) / (r2);            \
        s = (r2) + r * (i2);            \
        (r3) = ((r1) + r * (i1)) / s;       \
        (i3) = ((i1) - r * (r1)) / s;       \
    } else {                    \
        r = (r2) / (i2);            \
        s = (i2) + r * (r2);            \
        (r3) = (r * (r1) + (i1)) / s;       \
        (i3) = (r * (i1) - (r1)) / s;       \
    }                       \
}




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

/* Macro function that returns the approx absolute value of a complex
   number. */
#define  ELEMENT_MAG(ptr)   (ABS((ptr)->Real) + ABS((ptr)->Imag))

/* Complex assignment statements. */
#define  CMPLX_ASSIGN(to,from)  \
{   (to).Real = (from).Real;    \
    (to).Imag = (from).Imag;    \
}
#define  CMPLX_CONJ_ASSIGN(to,from)     \
{   (to).Real = (from).Real;            \
    (to).Imag = -(from).Imag;           \
}
#define  CMPLX_NEGATE_ASSIGN(to,from)   \
{   (to).Real = -(from).Real;           \
    (to).Imag = -(from).Imag;           \
}
#define  CMPLX_CONJ_NEGATE_ASSIGN(to,from)      \
{   (to).Real = -(from).Real;                   \
    (to).Imag = (from).Imag;                    \
}
#define  CMPLX_CONJ(a)  (a).Imag = -(a).Imag
#define  CMPLX_NEGATE(a)        \
{   (a).Real = -(a).Real;       \
    (a).Imag = -(a).Imag;       \
}

/* Macro that returns the approx magnitude (L-1 norm) of a complex number. */
#define  CMPLX_1_NORM(a)        (ABS((a).Real) + ABS((a).Imag))

/* Macro that returns the approx magnitude (L-infinity norm) of a complex. */
#define  CMPLX_INF_NORM(a)      (MAX (ABS((a).Real),ABS((a).Imag)))

/* Macro function that returns the magnitude (L-2 norm) of a complex number. */
#define  CMPLX_2_NORM(a)        (sqrt((a).Real*(a).Real + (a).Imag*(a).Imag))

/* Macro function that performs complex addition. */
#define  CMPLX_ADD(to,from_a,from_b)            \
{   (to).Real = (from_a).Real + (from_b).Real;  \
    (to).Imag = (from_a).Imag + (from_b).Imag;  \
}

/* Macro function that performs complex subtraction. */
#define  CMPLX_SUBT(to,from_a,from_b)           \
{   (to).Real = (from_a).Real - (from_b).Real;  \
    (to).Imag = (from_a).Imag - (from_b).Imag;  \
}

/* Macro function that is equivalent to += operator for complex numbers. */
#define  CMPLX_ADD_ASSIGN(to,from)      \
{   (to).Real += (from).Real;           \
    (to).Imag += (from).Imag;           \
}

/* Macro function that is equivalent to -= operator for complex numbers. */
#define  CMPLX_SUBT_ASSIGN(to,from)     \
{   (to).Real -= (from).Real;           \
    (to).Imag -= (from).Imag;           \
}

/* Macro function that multiplies a complex number by a scalar. */
#define  SCLR_MULT(to,sclr,cmplx)       \
{   (to).Real = (sclr) * (cmplx).Real;  \
    (to).Imag = (sclr) * (cmplx).Imag;  \
}

/* Macro function that multiply-assigns a complex number by a scalar. */
#define  SCLR_MULT_ASSIGN(to,sclr)      \
{   (to).Real *= (sclr);                \
    (to).Imag *= (sclr);                \
}

/* Macro function that multiplies two complex numbers. */
#define  CMPLX_MULT(to,from_a,from_b)           \
{   (to).Real = (from_a).Real * (from_b).Real - \
                (from_a).Imag * (from_b).Imag;  \
    (to).Imag = (from_a).Real * (from_b).Imag + \
                (from_a).Imag * (from_b).Real;  \
}

/* Macro function that implements to *= from for complex numbers. */
#define  CMPLX_MULT_ASSIGN(to,from)             \
{   RealNumber to_real_ = (to).Real;            \
    (to).Real = to_real_ * (from).Real -        \
                (to).Imag * (from).Imag;        \
    (to).Imag = to_real_ * (from).Imag +        \
                (to).Imag * (from).Real;        \
}

/* Macro function that multiplies two complex numbers, the first of which is
 * conjugated. */
#define  CMPLX_CONJ_MULT(to,from_a,from_b)      \
{   (to).Real = (from_a).Real * (from_b).Real + \
                (from_a).Imag * (from_b).Imag;  \
    (to).Imag = (from_a).Real * (from_b).Imag - \
                (from_a).Imag * (from_b).Real;  \
}

/* Macro function that multiplies two complex numbers and then adds them
 * to another. to = add + mult_a * mult_b */
#define  CMPLX_MULT_ADD(to,mult_a,mult_b,add)                   \
{   (to).Real = (mult_a).Real * (mult_b).Real -                 \
                (mult_a).Imag * (mult_b).Imag + (add).Real;     \
    (to).Imag = (mult_a).Real * (mult_b).Imag +                 \
                (mult_a).Imag * (mult_b).Real + (add).Imag;     \
}

/* Macro function that subtracts the product of two complex numbers from
 * another.  to = subt - mult_a * mult_b */
#define  CMPLX_MULT_SUBT(to,mult_a,mult_b,subt)                 \
{   (to).Real = (subt).Real - (mult_a).Real * (mult_b).Real +   \
                              (mult_a).Imag * (mult_b).Imag;    \
    (to).Imag = (subt).Imag - (mult_a).Real * (mult_b).Imag -   \
                              (mult_a).Imag * (mult_b).Real;    \
}

/* Macro function that multiplies two complex numbers and then adds them
 * to another. to = add + mult_a* * mult_b where mult_a* represents mult_a
 * conjugate. */
#define  CMPLX_CONJ_MULT_ADD(to,mult_a,mult_b,add)              \
{   (to).Real = (mult_a).Real * (mult_b).Real +                 \
                (mult_a).Imag * (mult_b).Imag + (add).Real;     \
    (to).Imag = (mult_a).Real * (mult_b).Imag -                 \
                (mult_a).Imag * (mult_b).Real + (add).Imag;     \
}

/* Macro function that multiplies two complex numbers and then adds them
 * to another. to += mult_a * mult_b */
#define  CMPLX_MULT_ADD_ASSIGN(to,from_a,from_b)        \
{   (to).Real += (from_a).Real * (from_b).Real -        \
                 (from_a).Imag * (from_b).Imag;         \
    (to).Imag += (from_a).Real * (from_b).Imag +        \
                 (from_a).Imag * (from_b).Real;         \
}

/* Macro function that multiplies two complex numbers and then subtracts them
 * from another. */
#define  CMPLX_MULT_SUBT_ASSIGN(to,from_a,from_b)       \
{   (to).Real -= (from_a).Real * (from_b).Real -        \
                 (from_a).Imag * (from_b).Imag;         \
    (to).Imag -= (from_a).Real * (from_b).Imag +        \
                 (from_a).Imag * (from_b).Real;         \
}

/* Macro function that multiplies two complex numbers and then adds them
 * to the destination. to += from_a* * from_b where from_a* represents from_a
 * conjugate. */
#define  CMPLX_CONJ_MULT_ADD_ASSIGN(to,from_a,from_b)   \
{   (to).Real += (from_a).Real * (from_b).Real +        \
                 (from_a).Imag * (from_b).Imag;         \
    (to).Imag += (from_a).Real * (from_b).Imag -        \
                 (from_a).Imag * (from_b).Real;         \
}

/* Macro function that multiplies two complex numbers and then subtracts them
 * from the destination. to -= from_a* * from_b where from_a* represents from_a
 * conjugate. */
#define  CMPLX_CONJ_MULT_SUBT_ASSIGN(to,from_a,from_b)  \
{   (to).Real -= (from_a).Real * (from_b).Real +        \
                 (from_a).Imag * (from_b).Imag;         \
    (to).Imag -= (from_a).Real * (from_b).Imag -        \
                 (from_a).Imag * (from_b).Real;         \
}

/*
 * Macro functions that provide complex division.
 */

/* Complex division:  to = num / den */
#define CMPLX_DIV(to,num,den)                                           \
{   RealNumber  r_, s_;                                                 \
    if (((den).Real >= (den).Imag AND (den).Real > -(den).Imag) OR      \
        ((den).Real < (den).Imag AND (den).Real <= -(den).Imag))        \
    {   r_ = (den).Imag / (den).Real;                                   \
        s_ = (den).Real + r_*(den).Imag;                                \
        (to).Real = ((num).Real + r_*(num).Imag)/s_;                    \
        (to).Imag = ((num).Imag - r_*(num).Real)/s_;                    \
    }                                                                   \
    else                                                                \
    {   r_ = (den).Real / (den).Imag;                                   \
        s_ = (den).Imag + r_*(den).Real;                                \
        (to).Real = (r_*(num).Real + (num).Imag)/s_;                    \
        (to).Imag = (r_*(num).Imag - (num).Real)/s_;                    \
    }                                                                   \
}

/* Complex division and assignment:  num /= den */
#define CMPLX_DIV_ASSIGN(num,den)                                       \
{   RealNumber  r_, s_, t_;                                             \
    if (((den).Real >= (den).Imag AND (den).Real > -(den).Imag) OR      \
        ((den).Real < (den).Imag AND (den).Real <= -(den).Imag))        \
    {   r_ = (den).Imag / (den).Real;                                   \
        s_ = (den).Real + r_*(den).Imag;                                \
        t_ = ((num).Real + r_*(num).Imag)/s_;                           \
        (num).Imag = ((num).Imag - r_*(num).Real)/s_;                   \
        (num).Real = t_;                                                \
    }                                                                   \
    else                                                                \
    {   r_ = (den).Real / (den).Imag;                                   \
        s_ = (den).Imag + r_*(den).Real;                                \
        t_ = (r_*(num).Real + (num).Imag)/s_;                           \
        (num).Imag = (r_*(num).Imag - (num).Real)/s_;                   \
        (num).Real = t_;                                                \
    }                                                                   \
}

/* Complex reciprocation:  to = 1.0 / den */
#define CMPLX_RECIPROCAL(to,den)                                        \
{   RealNumber  r_;                                                     \
    if (((den).Real >= (den).Imag && (den).Real > -(den).Imag) ||       \
        ((den).Real < (den).Imag && (den).Real <= -(den).Imag))         \
    {   r_ = (den).Imag / (den).Real;                                   \
        (to).Imag = -r_*((to).Real = 1.0/((den).Real + r_*(den).Imag)); \
    }                                                                   \
    else                                                                \
    {   r_ = (den).Real / (den).Imag;                                   \
        (to).Real = -r_*((to).Imag = -1.0/((den).Imag + r_*(den).Real));\
    }                                                                   \
}


#endif /*_COMPLEX_H */

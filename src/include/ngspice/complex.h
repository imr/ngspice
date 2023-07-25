/*
 * Copyright (c) 1985 Thomas L. Quarles
 * Modified: 1999 Paolo Nenzi, 2000 Arno W. Peters
 */
#ifndef ngspice_COMPLEX_H
#define ngspice_COMPLEX_H

/* Complex numbers. */
struct ngcomplex {
    double cx_real;
    double cx_imag;
} ;

typedef struct ngcomplex ngcomplex_t;

#define realpart(cval)  (cval).cx_real
#define imagpart(cval)  (cval).cx_imag

#ifdef CIDER
/* From Cider numcomplex.h 
   pn:leave it here until I decide what to do about 
struct mosAdmittances {
    ngcomplex_t yIdVdb;
    ngcomplex_t yIdVsb;
    ngcomplex_t yIdVgb;
    ngcomplex_t yIsVdb;
    ngcomplex_t yIsVsb;
    ngcomplex_t yIsVgb;
    ngcomplex_t yIgVdb;
    ngcomplex_t yIgVsb;
    ngcomplex_t yIgVgb;
   }; 
   */
#endif

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
 *  real  (realNumber)
 *      The real portion of the number.  real must be the first
 *      field in this structure.
 *  imag  (realNumber)
 *      The imaginary portion of the number. This field must follow
 *      immediately after real.
 */

#define spREAL  double

#if 0 /* Can this be deleted? */
 /* Begin `realNumber'. */
 typedef  spREAL  realNumber, *realVector;

 /* Begin `ComplexNumber'. */
 typedef  struct
 {   RealNumber  Real;
    RealNumber  Imag;
 } ComplexNumber, *ComplexVector;
#endif

/* Some defines used mainly in cmath.c. */
#define cph(c)      (atan2(imagpart(c), (realpart(c))))
#define cmag(c)     (hypot(realpart(c), imagpart(c)))
#define radtodeg(c) (cx_degrees ? ((c) * (180 / M_PI)) : (c))
#define degtorad(c) (cx_degrees ? ((c) * (M_PI / 180)) : (c))
#ifdef HAS_WINGUI
#define rcheck(cond, name)\
    if (!(cond)) {\
        (void) win_x_fprintf(cp_err, "Error: argument out of range for %s\n",\
                 name);\
        xrc = -1;\
        goto EXITPOINT;\
    }
#else
#define rcheck(cond, name)\
    if (!(cond)) {\
        (void) fprintf(cp_err, "Error: argument out of range for %s\n",\
                 name);\
        xrc = -1;\
        goto EXITPOINT;\
    }
#endif

#define cdiv(r1, i1, r2, i2, r3, i3)            \
{                           \
    double r, s;                    \
    if (fabs(r2) > fabs(i2)) {          \
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
	    _mag = hypot((A).real, (A).imag);                                 \
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
   
#define	C_ABS(A) (hypot((A).real, (A).imag))

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

#ifndef ngspice_SPDEFS_H

/* Macro function that returns the approx absolute value of a complex
   number. */
#define  ELEMENT_MAG(ptr)   (ABS((ptr)->real) + ABS((ptr)->imag))
 
#define  CMPLX_ASSIGN_VALUE(cnum, vreal, vimag)		\
{   (cnum).real = vreal;	\
    (cnum).imag = vimag;	\
}         

/* Complex assignment statements. */
#define  CMPLX_ASSIGN(to,from)  \
{   (to).real = (from).real;    \
    (to).imag = (from).imag;    \
}
#define  CMPLX_CONJ_ASSIGN(to,from)     \
{   (to).real = (from).real;            \
    (to).imag = -(from).imag;           \
}
#define  CMPLX_NEGATE_ASSIGN(to,from)   \
{   (to).real = -(from).real;           \
    (to).imag = -(from).imag;           \
}
#define  CMPLX_CONJ_NEGATE_ASSIGN(to,from)      \
{   (to).real = -(from).real;                   \
    (to).imag = (from).imag;                    \
}

#define  CMPLX_CONJ(a)  (a).imag = -(a).imag

#define  CONJUGATE(a)	(a).imag = -(a).imag

#define  CMPLX_NEGATE(a)        \
{   (a).real = -(a).real;       \
    (a).imag = -(a).imag;       \
}

#define  CMPLX_NEGATE_SELF(cnum)	\
{   (cnum).real = -(cnum).real;	\
    (cnum).imag = -(cnum).imag;	\
}

/* Macro that returns the approx magnitude (L-1 norm) of a complex number. */
#define  CMPLX_1_NORM(a)        (ABS((a).real) + ABS((a).imag))

/* Macro that returns the approx magnitude (L-infinity norm) of a complex. */
#define  CMPLX_INF_NORM(a)      (MAX (ABS((a).real),ABS((a).imag)))

/* Macro function that returns the magnitude (L-2 norm) of a complex number. */
#define  CMPLX_2_NORM(a)        (hypot((a).real, (a).imag))

/* Macro function that performs complex addition. */
#define  CMPLX_ADD(to,from_a,from_b)            \
{   (to).real = (from_a).real + (from_b).real;  \
    (to).imag = (from_a).imag + (from_b).imag;  \
}

/* Macro function that performs addition of a complex and a scalar. */
#define  CMPLX_ADD_SELF_SCALAR(cnum, scalar)      \
{   (cnum).real += scalar;   \
}

/* Macro function that performs complex subtraction. */
#define  CMPLX_SUBT(to,from_a,from_b)           \
{   (to).real = (from_a).real - (from_b).real;  \
    (to).imag = (from_a).imag - (from_b).imag;  \
}

/* Macro function that is equivalent to += operator for complex numbers. */
#define  CMPLX_ADD_ASSIGN(to,from)      \
{   (to).real += (from).real;           \
    (to).imag += (from).imag;           \
}

/* Macro function that is equivalent to -= operator for complex numbers. */
#define  CMPLX_SUBT_ASSIGN(to,from)     \
{   (to).real -= (from).real;           \
    (to).imag -= (from).imag;           \
}
 
/* Macro function that multiplies a complex number by a scalar. */
#define  SCLR_MULT(to,sclr,cmplx)       \
{   (to).real = (sclr) * (cmplx).real;  \
    (to).imag = (sclr) * (cmplx).imag;  \
}

/* Macro function that multiply-assigns a complex number by a scalar. */
#define  SCLR_MULT_ASSIGN(to,sclr)      \
{   (to).real *= (sclr);                \
    (to).imag *= (sclr);                \
}

/* Macro function that multiplies two complex numbers. */
#define  CMPLX_MULT(to,from_a,from_b)           \
{   (to).real = (from_a).real * (from_b).real - \
                (from_a).imag * (from_b).imag;  \
    (to).imag = (from_a).real * (from_b).imag + \
                (from_a).imag * (from_b).real;  \
}
 
/* Macro function that multiplies a complex number and a scalar. */
#define  CMPLX_MULT_SCALAR(to,from, scalar)      \
{   (to).real = (from).real * scalar;   \
    (to).imag = (from).imag * scalar;   \
}
 
/* Macro function that implements *= for a complex and a scalar number. */
 
#define  CMPLX_MULT_SELF_SCALAR(cnum, scalar)      \
{   (cnum).real *= scalar;   \
    (cnum).imag *= scalar;   \
}

/* Macro function that multiply-assigns a complex number by a scalar. */
#define  SCLR_MULT_ASSIGN(to,sclr)      \
{   (to).real *= (sclr);                \
    (to).imag *= (sclr);                \
}

/* Macro function that implements to *= from for complex numbers. */
#define  CMPLX_MULT_ASSIGN(to,from)             \
{   realNumber to_real_ = (to).real;            \
    (to).real = to_real_ * (from).real -        \
                (to).imag * (from).imag;        \
    (to).imag = to_real_ * (from).imag +        \
                (to).imag * (from).real;        \
}

/* Macro function that multiplies two complex numbers, the first of which is
 * conjugated. */
#define  CMPLX_CONJ_MULT(to,from_a,from_b)      \
{   (to).real = (from_a).real * (from_b).real + \
                (from_a).imag * (from_b).imag;  \
    (to).imag = (from_a).real * (from_b).imag - \
                (from_a).imag * (from_b).real;  \
}

/* Macro function that multiplies two complex numbers and then adds them
 * to another. to = add + mult_a * mult_b */
#define  CMPLX_MULT_ADD(to,mult_a,mult_b,add)                   \
{   (to).real = (mult_a).real * (mult_b).real -                 \
                (mult_a).imag * (mult_b).imag + (add).real;     \
    (to).imag = (mult_a).real * (mult_b).imag +                 \
                (mult_a).imag * (mult_b).real + (add).imag;     \
}

/* Macro function that subtracts the product of two complex numbers from
 * another.  to = subt - mult_a * mult_b */
#define  CMPLX_MULT_SUBT(to,mult_a,mult_b,subt)                 \
{   (to).real = (subt).real - (mult_a).real * (mult_b).real +   \
                              (mult_a).imag * (mult_b).imag;    \
    (to).imag = (subt).imag - (mult_a).real * (mult_b).imag -   \
                              (mult_a).imag * (mult_b).real;    \
}

/* Macro function that multiplies two complex numbers and then adds them
 * to another. to = add + mult_a* * mult_b where mult_a* represents mult_a
 * conjugate. */
#define  CMPLX_CONJ_MULT_ADD(to,mult_a,mult_b,add)              \
{   (to).real = (mult_a).real * (mult_b).real +                 \
                (mult_a).imag * (mult_b).imag + (add).real;     \
    (to).imag = (mult_a).real * (mult_b).imag -                 \
                (mult_a).imag * (mult_b).real + (add).imag;     \
}

/* Macro function that multiplies two complex numbers and then adds them
 * to another. to += mult_a * mult_b */
#define  CMPLX_MULT_ADD_ASSIGN(to,from_a,from_b)        \
{   (to).real += (from_a).real * (from_b).real -        \
                 (from_a).imag * (from_b).imag;         \
    (to).imag += (from_a).real * (from_b).imag +        \
                 (from_a).imag * (from_b).real;         \
}

/* Macro function that multiplies two complex numbers and then subtracts them
 * from another. */
#define  CMPLX_MULT_SUBT_ASSIGN(to,from_a,from_b)       \
{   (to).real -= (from_a).real * (from_b).real -        \
                 (from_a).imag * (from_b).imag;         \
    (to).imag -= (from_a).real * (from_b).imag +        \
                 (from_a).imag * (from_b).real;         \
}

/* Macro function that multiplies two complex numbers and then adds them
 * to the destination. to += from_a* * from_b where from_a* represents from_a
 * conjugate. */
#define  CMPLX_CONJ_MULT_ADD_ASSIGN(to,from_a,from_b)   \
{   (to).real += (from_a).real * (from_b).real +        \
                 (from_a).imag * (from_b).imag;         \
    (to).imag += (from_a).real * (from_b).imag -        \
                 (from_a).imag * (from_b).real;         \
}

/* Macro function that multiplies two complex numbers and then subtracts them
 * from the destination. to -= from_a* * from_b where from_a* represents from_a
 * conjugate. */
#define  CMPLX_CONJ_MULT_SUBT_ASSIGN(to,from_a,from_b)  \
{   (to).real -= (from_a).real * (from_b).real +        \
                 (from_a).imag * (from_b).imag;         \
    (to).imag -= (from_a).real * (from_b).imag -        \
                 (from_a).imag * (from_b).real;         \
}

/*
 * Macro functions that provide complex division.
 */

/* Complex division:  to = num / den */
#define CMPLX_DIV(to,num,den)                                           \
{   realNumber  r_, s_;                                                 \
    if (((den).real >= (den).imag && (den).real > -(den).imag) ||       \
        ((den).real < (den).imag && (den).real <= -(den).imag))         \
    {   r_ = (den).imag / (den).real;                                   \
        s_ = (den).real + r_*(den).imag;                                \
        (to).real = ((num).real + r_*(num).imag)/s_;                    \
        (to).imag = ((num).imag - r_*(num).real)/s_;                    \
    }                                                                   \
    else                                                                \
    {   r_ = (den).real / (den).imag;                                   \
        s_ = (den).imag + r_*(den).real;                                \
        (to).real = (r_*(num).real + (num).imag)/s_;                    \
        (to).imag = (r_*(num).imag - (num).real)/s_;                    \
    }                                                                   \
}

/* Complex division and assignment:  num /= den */
#define CMPLX_DIV_ASSIGN(num,den)                                       \
{   realNumber  r_, s_, t_;                                             \
    if (((den).real >= (den).imag && (den).real > -(den).imag) ||       \
        ((den).real < (den).imag && (den).real <= -(den).imag))         \
    {   r_ = (den).imag / (den).real;                                   \
        s_ = (den).real + r_*(den).imag;                                \
        t_ = ((num).real + r_*(num).imag)/s_;                           \
        (num).imag = ((num).imag - r_*(num).real)/s_;                   \
        (num).real = t_;                                                \
    }                                                                   \
    else                                                                \
    {   r_ = (den).real / (den).imag;                                   \
        s_ = (den).imag + r_*(den).real;                                \
        t_ = (r_*(num).real + (num).imag)/s_;                           \
        (num).imag = (r_*(num).imag - (num).real)/s_;                   \
        (num).real = t_;                                                \
    }                                                                   \
}

/* Complex reciprocation:  to = 1.0 / den */
#define CMPLX_RECIPROCAL(to,den)                                        \
{   realNumber  r_;                                                     \
    if (((den).real >= (den).imag && (den).real > -(den).imag) ||       \
        ((den).real < (den).imag && (den).real <= -(den).imag))         \
    {   r_ = (den).imag / (den).real;                                   \
        (to).imag = -r_*((to).real = 1.0/((den).real + r_*(den).imag)); \
    }                                                                   \
    else                                                                \
    {   r_ = (den).real / (den).imag;                                   \
        (to).real = -r_*((to).imag = -1.0/((den).imag + r_*(den).real));\
    }                                                                   \
}

#endif /* ngspice_SPDEF_H */

#endif

/*************
 * Macros definitions header file
 * 1999 E. Rouat
 ************/

/* 
 * This file will contain all macros definitions needed
 * by ngspice code. (in construction)
 */


#ifndef _MACROS_H_
#define _MACROS_H_

#define	NUMELEMS(ARRAY)	(sizeof(ARRAY)/sizeof(*ARRAY))

/*
 * Macros for complex mathematical functions.
 */

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




#define tfree(x)	(txfree(x), x = 0)
#define	alloc(TYPE)	((TYPE *) tmalloc(sizeof(TYPE)))


#define eq(a,b)  (!strcmp((a), (b)))
#define eqc(a,b)  (cieq((a), (b)))
#define isalphanum(c)   (isalpha(c) || isdigit(c))
#define hexnum(c) ((((c) >= '0') && ((c) <= '9')) ? ((c) - '0') : ((((c) >= \
        'a') && ((c) <= 'f')) ? ((c) - 'a' + 10) : ((((c) >= 'A') && \
        ((c) <= 'F')) ? ((c) - 'A' + 10) : 0)))

#define MALLOC(x) tmalloc((unsigned)(x))
#define FREE(x) {if (x) {free((char *)(x));(x) = 0;}}
#define REALLOC(x,y) trealloc((char *)(x),(unsigned)(y))
#define ZERO(PTR,TYPE)	(bzero((PTR),sizeof(TYPE)))


#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define SIGN(a,b) ( b >= 0 ? (a >= 0 ? a : - a) : (a >= 0 ? - a : a))

#define ABORT() fflush(stderr);fflush(stdout);abort();

#define ERROR(CODE,MESSAGE)	{					      \
	errMsg = MALLOC(strlen(MESSAGE) + 1);				      \
	strcpy(errMsg, (MESSAGE));					      \
	return (CODE);							      \
	}

#define	NEW(TYPE)	((TYPE *) MALLOC(sizeof(TYPE)))
#define	NEWN(TYPE,COUNT) ((TYPE *) MALLOC(sizeof(TYPE) * (COUNT)))


#define	R_NORM(A,B) {							      \
	if ((A) == 0.0) {						      \
	    (B) = 0;							      \
	} else {							      \
	    while (fabs(A) > 1.0) {					      \
		(B) += 1;						      \
		(A) /= 2.0;						      \
	    }								      \
	    while (fabs(A) < 0.5) {					      \
		(B) -= 1;						      \
		(A) *= 2.0;						      \
	    }								      \
	}								      \
    }


#ifdef DEBUG
#define DEBUGMSG(textargs) printf(textargs)
#else
#define DEBUGMSG(testargs) 
#endif


#define realpart(cval)  ((struct _complex *) (cval))->cx_real
#define imagpart(cval)  ((struct _complex *) (cval))->cx_imag



#endif /* _MACROS_H_ */

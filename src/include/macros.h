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

#define eq(a,b)  (!strcmp((a), (b)))
#define eqc(a,b)  (cieq((a), (b)))
#define isalphanum(c)   (isalpha(c) || isdigit(c))
#define hexnum(c) ((((c) >= '0') && ((c) <= '9')) ? ((c) - '0') : ((((c) >= \
        'a') && ((c) <= 'f')) ? ((c) - 'a' + 10) : ((((c) >= 'A') && \
        ((c) <= 'F')) ? ((c) - 'A' + 10) : 0)))

#define tfree(x) (txfree(x), x = 0)
#define alloc(TYPE) ((TYPE *) tmalloc(sizeof(TYPE)))
#define MALLOC(x) tmalloc((unsigned)(x))
#define FREE(x) {if (x) {txfree((char *)(x));(x) = 0;}}
#define REALLOC(x,y) trealloc((char *)(x),(unsigned)(y))
#define ZERO(PTR,TYPE)	(bzero((PTR),sizeof(TYPE)))


#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define SIGN(a,b) ( b >= 0 ? (a >= 0 ? a : - a) : (a >= 0 ? - a : a))

#define ABORT() fflush(stderr);fflush(stdout);abort();

#define ERROR(CODE,MESSAGE)	{			      \
	errMsg = (char *) tmalloc(strlen(MESSAGE) + 1);     \
	strcpy(errMsg, (MESSAGE));			      \
	return (CODE);					      \
	}

#define	NEW(TYPE)	((TYPE *) tmalloc(sizeof(TYPE)))
#define	NEWN(TYPE,COUNT) ((TYPE *) tmalloc(sizeof(TYPE) * (COUNT)))


#define	R_NORM(A,B) {					      \
	if ((A) == 0.0) {				      \
	    (B) = 0;					      \
	} else {					      \
	    while (fabs(A) > 1.0) {			      \
		(B) += 1;				      \
		(A) /= 2.0;				      \
	    }						      \
	    while (fabs(A) < 0.5) {			      \
		(B) -= 1;				      \
		(A) *= 2.0;				      \
	    }						      \
	}						      \
    }


#ifdef DEBUG
#define DEBUGMSG(textargs) printf(textargs)
#else
#define DEBUGMSG(testargs) 
#endif

#endif /* _MACROS_H_ */

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

/* String macros */

#define eq(a,b)  (!strcmp((a), (b)))
#define eqc(a,b)  (cieq((a), (b)))
#define isalphanum(c)   (isalpha(c) || isdigit(c))
#define hexnum(c) ((((c) >= '0') && ((c) <= '9')) ? ((c) - '0') : ((((c) >= \
        'a') && ((c) <= 'f')) ? ((c) - 'a' + 10) : ((((c) >= 'A') && \
        ((c) <= 'F')) ? ((c) - 'A' + 10) : 0)))


/* Mathematical macros */

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define ABS(a)    ((a) < 0.0 ? -(a) : (a))
#define SGN(a)    ((a) < 0.0 ? -(1.0) : (1.0))
#define SIGN(a,b) ( b >= 0 ? (a >= 0 ? a : - a) : (a >= 0 ? - a : a))
#define  SWAP(type, a, b)   {type swapx; swapx = a; a = b; b = swapx;}
 
 
#define NIL(type) ((type *)0) 
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

/* Macro that queries the system to find the process time. */
 
#define ELAPSED_TIME( time )				\
{   struct tms {int user, sys, cuser, csys;} buffer;	\
 							\
    times(&buffer);					\
    time = buffer.user / 60.0;				\
}

#ifdef HAVE_SIGSETJMP
# define SETJMP(env, save_signals) sigsetjmp(env, save_signals)
# define LONGJMP(env, retval) siglongjmp(env, retval)
# define JMP_BUF sigjmp_buf
#else
# define SETJMP(env, save_signals) setjmp(env)
# define LONGJMP(env, retval) longjmp(env, retval)
# define JMP_BUF jmp_buf
#endif


#endif /* _MACROS_H_ */

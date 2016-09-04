/*************
 * Macros definitions header file
 * 1999 E. Rouat
 ************/

/* 
 * This file will contain all macros definitions needed
 * by ngspice code. (in construction)
 */


#ifndef ngspice_MACROS_H
#define ngspice_MACROS_H

/*
 *  #define-s that are always on
 */

#define NEWCONV

#define	NUMELEMS(ARRAY)	(sizeof(ARRAY)/sizeof(*ARRAY))

/* String macros */

#define eq(a,b)  (!strcmp((a), (b)))
#define eqc(a,b)  (cieq((a), (b)))
#define hexnum(c) ((((c) >= '0') && ((c) <= '9')) ? ((c) - '0') : ((((c) >= \
        'a') && ((c) <= 'f')) ? ((c) - 'a' + 10) : ((((c) >= 'A') && \
        ((c) <= 'F')) ? ((c) - 'A' + 10) : 0)))


/* Mathematical macros */

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define ABS(a)    ((a) < 0.0 ? -(a) : (a))
#define SGN(a)    copysign(1.0, (a))
#define SWAP(type, a, b)                        \
    do {                                        \
        type SWAP_macro_local = a;              \
        a = b;                                  \
        b = SWAP_macro_local;                   \
    } while(0)
 
 
#define ABORT()                                 \
    do {                                        \
        fflush(stderr);                         \
        fflush(stdout);                         \
        abort();                                \
    } while(0)

#define MERROR(CODE, MESSAGE) \
    do {                                                      \
        errMsg = TMALLOC(char, strlen(MESSAGE) + 1);          \
        strcpy(errMsg, (MESSAGE));                            \
        return (CODE);                                        \
    } while(0)


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
#define DS(name_xz)  { name_xz }
#define DBGDEFINE(func_xz)  func_xz
#else  /* ! DEBUG */
#define DEBUGMSG(testargs) 
#define DS(name_xz)
#define DBGDEFINE(func_xz)  
#endif /* DEBUG */

/* A few useful macros - string eq just makes the code easier to read */
#define STRINGEQ 0
#define FUNC_NAME(x_xz) char *routine = x_xz

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


#endif

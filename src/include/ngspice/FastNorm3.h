/*	Last revised 28-1-1999	*/
/*	This is the header file FastNorm3.h to be included in code files
	using FastNorm3.c	*/
/*	                      I M P O R T A N T   ! ! ! ! !

	The definition below should be altered to ensure that integer
arithmetic is done on 32-bit words. It may need to be changed from int to
long on some platforms.  The 32-bit requirement arises from the use of
a Uniform pseudo-random generator in part of the code, which assumes 32-bit
twos-complement arithmetic.  In dire need, replace this generator with
another more suitable for the platform. The rest of the code assumes only
that signed integers up to a bit less than 2^31 can be handled.
	*/

#define Sw int		/* MUST define Sw as a 32-bit integer or longer */
#define Sf double

extern int gaussfaze;
extern int gaussmask;
extern double *gausssave;
extern double GScale;

#define FastNorm ((--gaussfaze)?GScale*gausssave[gaussfaze]:fastnorm())

void initnorm(Sw seed, Sw quoll);
Sf fastnorm (void);
Sf c7rand(Sw*);
Sw irandm(Sw*);
unsigned Sw urandm(Sw*);
double adchi (double a, int *is);
double rgamma (double g, int *is);
Sf renormalize(void);

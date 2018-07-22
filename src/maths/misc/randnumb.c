/**********
Copyright 2008 Holger Vogt
**********/
/* Care about random numbers
   The seed value is set as random number in main.c, line 746.
   A fixed seed value may be set by 'set rndseed=value'.
*/


/*
	CombLCGTaus()
	Combined Tausworthe & LCG random number generator
	Algorithm has been suggested in: 
	GPUGems 3, Addison Wesley, 2008, Chapter 37.
	It combines a three component Tausworthe generator taus88 
	(see P. L’Ecuyer: "Maximally equidistributed combined Tausworthe
	generators", Mathematics of Computation, 1996, 
	http://www.iro.umontreal.ca/~lecuyer/myftp/papers/tausme.ps )
	and a quick linear congruent generator (LCG), decribed in:
	Press: "Numerical recipes in C", Cambridge, 1992, p. 284.
	Generator has passed the bbattery_SmallCrush(gen) test of the
	TestU01 library from Pierre L’Ecuyer and Richard Simard,
	http://www.iro.umontreal.ca/~simardr/testu01/tu01.html
*/


/* TausSeed creates three start values for Tausworthe state variables.
   Uses rand() from <stdlib.h>, therefore values depend on the value of 
   seed in srand(seed). A constant seed will result in a reproducible
   series of random variates.
   
   Calling sequence:
   srand(seed);
   TausSeed();
   double randvar = CombLCGTaus(void);
*/
//#define HVDEBUG

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/randnumb.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef _MSC_VER
#include <process.h>
#else
#include <unistd.h>
#endif
#include <stdarg.h>			// var. argumente

/* Tausworthe state variables for double variates*/
static unsigned CombState1 = 129, CombState2 = 130, CombState3 = 131;  
static unsigned CombState4 = 132; /* LCG state variable */ 

/* Tausworthe state variables for integer variates*/
static unsigned CombState5 = 133, CombState6 = 135, CombState7 = 137;  
static unsigned CombState8 = 138; /* LCG state variable */ 

static unsigned TauS(unsigned *state, int C1, int C2, int C3, unsigned m);
static unsigned LGCS(unsigned *state, unsigned A1, unsigned A2);

double CombLCGTaus(void);
float  CombLCGTaus2(void);

void rgauss(double* py1, double* py2);
static bool seedinfo = FALSE;


/* Check if a seed has been set by the command 'set rndseed=value'
   in spinit, .spiceinit or in a .control section
   with integer value > 0. If available, call srand(value).
   rndseed set in main.c to 1, if no 'set rndseed=val' is given.
   Called from functions in cmath2.c.
*/
void checkseed(void)
{
   int newseed;
   static int oldseed;
/*   printf("Enter checkseed()\n"); */
   if (cp_getvar("rndseed", CP_NUM, &newseed, 0)) {
      if ((newseed > 0) && (oldseed != newseed)) {
         srand((unsigned int)newseed);
         TausSeed();
         if (oldseed > 0) /* no printout upon start-up */
             printf("Seed value for random number generator is set to %d\n", newseed);
         oldseed = newseed;
      }
   }
   
}

/* uniform random number generator, interval [-1 .. +1[ */
double drand(void)
{
   return 2.0 * CombLCGTaus() - 1.0;
}


void TausSeed(void)
{    
   /* The Tausworthe initial states should be greater than 128.
      We restrict the values up to 32767. 
      Here we use the standard random functions srand, called in main.c
      upon ngspice startup or later in fcn checkseed(),
      rand() and the maximum return value RAND_MAX*/
   CombState1 = (unsigned int)((double)rand()/(double)RAND_MAX * 32638.) + 129;
   CombState2 = (unsigned int)((double)rand()/(double)RAND_MAX * 32638.) + 129;
   CombState3 = (unsigned int)((double)rand()/(double)RAND_MAX * 32638.) + 129;
   CombState4 = (unsigned int)((double)rand()/(double)RAND_MAX * 32638.) + 129;
   CombState5 = (unsigned int)((double)rand()/(double)RAND_MAX * 32638.) + 129;
   CombState6 = (unsigned int)((double)rand()/(double)RAND_MAX * 32638.) + 129;
   CombState7 = (unsigned int)((double)rand()/(double)RAND_MAX * 32638.) + 129;
   CombState8 = (unsigned int)((double)rand()/(double)RAND_MAX * 32638.) + 129;

#ifdef HVDEBUG
   printf("\nTausworthe Double generator init states: %d, %d, %d, %d\n", 
      CombState1, CombState2, CombState3, CombState4);
   printf("Tausworthe Integer generator init states: %d, %d, %d, %d\n", 
      CombState5, CombState6, CombState7, CombState8);
#endif
}   

static unsigned TauS(unsigned *state, int C1, int C2, int C3, unsigned m)
{
   unsigned b = (((*state << C1) ^ *state) >> C2);
   return *state = (((*state & m) << C3) ^ b);
}

static unsigned LGCS(unsigned *state, unsigned A1, unsigned A2)
{
   return *state = (A1 * *state + A2);
}

/* generate random variates randvar uniformly distributed in 
   [0.0 .. 1.0[ by calls to CombLCGTaus() like:
   double randvar = CombLCGTaus(); 
*/
double CombLCGTaus(void)
{
   return 2.3283064365387e-10 * (
   TauS(&CombState1, 13, 19, 12, 4294967294UL) ^
   TauS(&CombState2, 2, 25, 4, 4294967288UL) ^
   TauS(&CombState3, 3, 11, 17, 4294967280UL) ^
   LGCS(&CombState4, 1664525, 1013904223UL)
   );
}

/* generate random variates randvarint uniformly distributed in 
   [0 .. 4294967296[ (32 bit unsigned int) by calls to CombLCGTausInt() like:
   unsigned int randvarint = CombLCGTausInt(); 
*/   
unsigned int CombLCGTausInt(void)
{
   return (
   TauS(&CombState5, 13, 19, 12, 4294967294UL) ^
   TauS(&CombState6, 2, 25, 4, 4294967288UL) ^
   TauS(&CombState7, 3, 11, 17, 4294967280UL) ^
   LGCS(&CombState8, 1664525, 1013904223UL)
   );
}
  
/* test versions of the generators listed above */
float CombLCGTaus2(void)
{
   unsigned long b;
   b = (((CombState1 << 13) ^ CombState1) >> 19);
   CombState1 = (unsigned int)(((CombState1 & 4294967294UL) << 12) ^ b);
   b = (((CombState2 << 2) ^ CombState2) >> 25);
   CombState2 = (unsigned int)(((CombState2 & 4294967288UL) << 4) ^ b);   
   b = (((CombState3 << 3) ^ CombState3) >> 11);
   CombState3 = (unsigned int)(((CombState3 & 4294967280UL) << 17) ^ b);
   CombState4 = (unsigned int)(1664525 * CombState4 + 1013904223UL);   
   return ((float)(CombState1 ^ CombState2 ^ CombState3 ^ CombState4) *  2.3283064365387e-10f);
}


unsigned int CombLCGTausInt2(void)
{
   unsigned long b;
   b = (((CombState5 << 13) ^ CombState5) >> 19);
   CombState5 = (unsigned int)(((CombState5 & 4294967294UL) << 12) ^ b);
   b = (((CombState6 << 2) ^ CombState6) >> 25);
   CombState6 = (unsigned int)(((CombState6 & 4294967288UL) << 4) ^ b);   
   b = (((CombState7 << 3) ^ CombState7) >> 11);
   CombState7 = (unsigned int)(((CombState7 & 4294967280UL) << 17) ^ b);
   CombState8 = (unsigned int)(1664525 * CombState8 + 1013904223UL);   
   return (CombState5 ^ CombState6 ^ CombState7 ^ CombState8);
}


/***  gauss  ***
 for speed reasons get two values per pass */
double gauss0(void)
{
  static bool gliset = TRUE;
  static double glgset = 0.0;
  double fac,r,v1,v2;
  if (gliset) {
    do {
      v1 = 2.0 * CombLCGTaus() - 1.0;
      v2 = 2.0 * CombLCGTaus() - 1.0;
      r = v1*v1 + v2*v2;
    } while (r >= 1.0);
/*    printf("v1 %f, v2 %f\n", v1, v2); */
    fac = sqrt(-2.0 * log(r) / r);
    glgset = v1 * fac;
    gliset = FALSE;
    return v2 * fac;
  } else {
    gliset = TRUE;
    return glgset;
  }
}


/***  gauss  ***
to be reproducible, we just use one value per pass */
double gauss1(void)
{
    double fac, r, v1, v2;
    do {
        v1 = 2.0 * CombLCGTaus() - 1.0;
        v2 = 2.0 * CombLCGTaus() - 1.0;
        r = v1 * v1 + v2 * v2;
    } while (r >= 1.0);
    /*    printf("v1 %f, v2 %f\n", v1, v2); */
    fac = sqrt(-2.0 * log(r) / r);
    return v2 * fac;
}


/* Polar form of the Box-Muller generator for Gaussian distributed
   random variates.
   Generator will be fed with two uniformly distributed random variates.
   Delivers two values per call
*/

void rgauss(double* py1, double* py2)
{
    double x1, x2, w;

    do {
        x1 = 2.0 * CombLCGTaus() - 1.0;
        x2 = 2.0 * CombLCGTaus() - 1.0;
        w = x1 * x1 + x2 * x2;
    } while ( w >= 1.0 );

     w = sqrt( (-2.0 * log( w ) ) / w );

    *py1 = x1 * w;
    *py2 = x2 * w;
}



/** Code by: Inexpensive
    http://everything2.com/title/Generating+random+numbers+with+a+Poisson+distribution **/
int poisson(double lambda)
{
  int k=0;                          //Counter
  const int max_k = 1000;           //k upper limit
  double p = CombLCGTaus();         //uniform random number
  double P = exp(-lambda);        //probability
  double sum=P;                     //cumulant
  if (sum>=p) return 0;             //done allready
  for (k=1; k<max_k; ++k) {         //Loop over all k:s
    P*=lambda/(double)k;           //Calc next prob
    sum+=P;                         //Increase cumulant
    if (sum>=p) break;              //Leave loop
  }
  return k;                         //return random number
}


/* return an exponentially distributed random number */
double exprand(double mean)
{
    double expval;
    expval = -log(CombLCGTaus()) * mean;
    return expval;
}


/* seed random number generators immediately
* command "setseed"
*   take value of variable rndseed as seed
* command "setseed <n>"
*   seed with number <n>
*/
void
com_sseed(wordlist *wl)
{
    int newseed;

    if (wl == NULL) {
        if (!cp_getvar("rndseed", CP_NUM, &newseed, 0)) {
            newseed = getpid();
            cp_vset("rndseed", CP_NUM, &newseed);
        }
        srand((unsigned int)newseed);
        TausSeed();
    }
    else if ((sscanf(wl->wl_word, " %d ", &newseed) != 1) ||
        (newseed <= 0) || (newseed > INT_MAX))
    {
        fprintf(cp_err,
            "\nWarning: Cannot use %s as seed!\n"
            "    Command 'setseed %s' ignored.\n\n",
            wl->wl_word, wl->wl_word);
        return;
    }
    else {
        srand((unsigned int)newseed);
        TausSeed();
        cp_vset("rndseed", CP_NUM, &newseed);
    }

    if (seedinfo)
        printf("\nSeed value for random number generator is set to %d\n", newseed);
}


void
setseedinfo(void)
{
    seedinfo = TRUE;
}

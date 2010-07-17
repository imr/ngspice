/**********
Copyright 2008 Holger Vogt
**********/
/* Care about random numbers
   The seed value is set as random number in main.c, line 746.
   A fixed seed value may be set by 'set rndseed=value'.
*/

#include "ngspice.h"
#include "cpdefs.h"
#include "ftedefs.h"


/* MINGW: random, srandom in libiberty.a, but not in libiberty.h */
#if defined(__MINGW32__) && defined(HAVE_RANDOM)
extern long int random (void);
extern void srandom (unsigned int seed);
#endif

void checkseed(void);
double drand(void);
double gauss(void);


/* Check if a seed has been set by the command 'set rndseed=value'
   in spinit with integer value > 0. If available, call srand(value).
   This will override the call to srand in main.c.
   Checkseed should be put in front of any call to random or rand.
*/
void checkseed(void)
{
   int newseed;
   static int oldseed;
/*   printf("Enter checkseed()\n"); */
   if (cp_getvar("rndseed", CP_NUM, &newseed)) {
      if ((newseed > 0) && (oldseed != newseed)) {
         srandom(newseed);
         oldseed = newseed;
         printf("Seed value for random number generator is set to %d\n", newseed);
      }
/*      else printf("Oldseed %d, newseed %d\n", oldseed, newseed); */
   }
   
}

/* uniform random number generator, interval -1 .. +1 */
double drand(void)
{
   checkseed();
   return ( 2.0*((double) (RR_MAX-abs(random())) / (double)RR_MAX-0.5));
}


/***  gauss  ***/

double gauss(void)
{
  static bool gliset = TRUE;
  static double glgset = 0.0;
  double fac,r,v1,v2;
  if (gliset) {
    do {
      v1 = drand();  v2 = drand();
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




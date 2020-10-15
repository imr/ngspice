/* Wallace generator for normally distributed random variates
   Copyright Holger Vogt, 2008
   
   Calling sequence:
   initw(void); initialize using srand(seed)
   double x = GaussWa;  returns normally distributed random variate
   
*/   



extern double *outgauss; /*output vector for user access */
extern unsigned int variate_used; /* actual index of variate called by user */
extern double ScaleGauss; /* scale factor, including chi square correction */

double NewWa(void); /* generate new pool, return outgauss[0] */

#define GaussWa ((--variate_used)?(outgauss[variate_used]*ScaleGauss):NewWa())

void PolarGauss(double* py1, double* py2);

void destroy_wallace(void);

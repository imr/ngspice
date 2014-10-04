#include <math.h>
void cm_limitsmbv(ARGS)  /* structure holding parms,
                                       inputs, outputs, etc.     */
{

    double x;			/* Input Variable */
    double xtp;			/* Transition threshold from linear to exponention (positive side) */
    double xtm;			/* Transition threshold from linear to exponention (positive side) */

    double y;			/* Output Variable */

    double xtpi;		/* xtp limited to positive range */
    double xtmi;		/* xtm limited to negative range */

    double g;			/* Linear slope passing through x,y=0,0 */
    double a;			/* Term for exponetial regions below xtm and above xtp */
    double Xg;			/* Relation for continuous derivatives at xtm and xtp */
    double Xo;			/* Offset of exponentional terms for continuos derivatives */

    Mif_Complex_t dy_dx;	/* Complex partial derivative dy/dx */
    Mif_Complex_t dy_dxtp;	/* Complex partial derivative dy/dxtp */
    Mif_Complex_t dy_dxtm;	/* Complex partial derivative dy/dxtm */


   /*
    * Linear to Exponention Transistion Management
    */
    g =			PARAM(g);
    a =			PARAM(a);
    if ( g > 0.0 ) {
      Xg =		log( g / a ) / a;
      Xo =		exp( a * Xg );
    }
    else {
      Xg =		0.0;
      Xo =		1.0;
    }

   /*
    * Access the inputs from the interface
    */
    x =			INPUT(x);
    xtp =		INPUT(xtp);
    xtm =		INPUT(xtm);


   /*
    * Sanity check on dynamic thresholds.
    */
    xtpi =		(xtp >= 0.0) ? xtp : 0.0;
    xtmi =		(xtm <= 0.0) ? xtm : 0.0;

   /*
    * Ideal Avalanche Characteristic
    */
    if (x >= xtpi) {
      y			= g * xtpi+exp(a * (x - xtpi+Xg)) - Xo;
      dy_dx.real	= a*exp(a * (x - xtpi+Xg));
      dy_dxtp.real	= g - a*exp(a * (x - xtpi+Xg));
      dy_dxtm.real	= 0.0;
    }
    else if (x <= xtmi) {
      y			= -1.0 * (g * fabs(xtmi)+exp(a * (fabs(x) - fabs(xtmi)+Xg)) - Xo);
      dy_dx.real	= -1.0*a*exp(a * (fabs(x) - fabs(xtmi)+Xg));;
      dy_dxtp.real	= 0.0;
      dy_dxtm.real	= -1.0*(g - a*exp(a * (fabs(x) - fabs(xtmi)+Xg)));
    }
    else {
      y			= g * x;
      dy_dx.real	= g;
      dy_dxtp.real	= 0.0;
      dy_dxtm.real	= 0.0;
    }

   /*
    * Zero out the imaginary parts.
    */
    dy_dx.imag		= 0.0;
    dy_dxtp.imag	= 0.0;
    dy_dxtm.imag	= 0.0;


    if (ANALYSIS != MIF_AC) {		/* DC & Transient Analyses */
      OUTPUT(y) =	y;
      PARTIAL(y,x) =	dy_dx.real;
      PARTIAL(y,xtp) =	dy_dxtp.real;
      PARTIAL(y,xtm) =	dy_dxtm.real;
    }
    else {				/* AC Analysis */
      AC_GAIN(y,x) =	dy_dx;
      AC_GAIN(y,xtp) =	dy_dxtp;
      AC_GAIN(y,xtm) =	dy_dxtm;
    }
}

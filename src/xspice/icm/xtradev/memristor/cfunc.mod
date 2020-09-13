/* ===========================================================================
FILE    memristor/cfunc.mod

MEMBER OF process XSPICE

-------------------------------------------------------------------------
 Copyright 2012
 The ngspice team
 All Rights Reserved
 3 - Clause BSD license
 (see COPYING or https://opensource.org/licenses/BSD-3-Clause)
-------------------------------------------------------------------------


AUTHORS

    6/08/2012  Holger Vogt

MODIFICATIONS

    <date> <person name> <nature of modifications>

SUMMARY

    This file contains the definition of a memristor code model
    with threshold according to
	Y. V. Pershin, M. Di Ventra: "SPICE model of memristive devices with threshold", 
    arXiv:1204.2600v1 [physics.comp-ph] 12 Apr 2012, 
    http://arxiv.org/pdf/1204.2600.pdf.
	
	** Experimental, still to be tested in circuits !! **

	dc and ac simulation just return rinit.

INTERFACES

    cm_memristor()

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

=========================================================================== */

/*=== INCLUDE FILES ====================*/

#include <math.h>


#define RV  0

/* model parameters */
double alpha, beta, vt;
/* forward of window function */
double f1(double y); 

void cm_memristor (ARGS)
{
    Complex_t   ac_gain;
    double      partial;
    double      int_value;
    double      *rval;
	double      inpdiff;
	
    /* get the parameters */
	alpha = PARAM(alpha);
    beta = PARAM(beta);
    vt = PARAM(vt);	

    /* Initialize/access instance specific storage for resistance value */
    if(INIT) {
        cm_analog_alloc(RV, sizeof(double));
        rval = (double *) cm_analog_get_ptr(RV, 0);
        *rval = PARAM(rinit);
    }
    else {
        rval = (double *) cm_analog_get_ptr(RV, 0);
    }

    /* Compute the output */
    if(ANALYSIS == TRANSIENT) {
	    /* input the voltage across the terminals */
	    inpdiff = f1(INPUT(memris));
	    if ((inpdiff > 0) && (*rval < PARAM(rmax)))
		    int_value = inpdiff;
		else if  ((inpdiff < 0) && (*rval > PARAM(rmin)))
		    int_value = inpdiff;
        else
		    int_value = 0.0;
		/* integrate the new resistance */	
        cm_analog_integrate(int_value, rval, &partial);
		/* output the current */
        OUTPUT(memris) = INPUT(memris) / *rval;
        /* This does work, but is questionable */
        PARTIAL(memris, memris) = partial;
        /* This may be a (safe?) replacement, but in fact is not
        so good	at high voltage	(at strong non-linearity)
		cm_analog_auto_partial();*/
    }
    else if(ANALYSIS == AC) {
        ac_gain.real = 1/ *rval;
        ac_gain.imag = 0.0;
        AC_GAIN(memris, memris) = ac_gain;
    }
	else
	    OUTPUT(memris) = INPUT(memris) / *rval;
}

/* the window function */
double f1(double y) {
    return (beta*y+0.5*(alpha-beta)*(fabs(y+vt)-fabs(y-vt)));
}	



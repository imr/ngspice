/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE divide/cfunc.mod

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503-405
               

AUTHORS                      

    6 Jun 1991     Jeffrey P. Murray


MODIFICATIONS   

     2 Oct 1991    Jeffrey P. Murray
                                   

SUMMARY

    This file contains the model-specific routines used to
    functionally describe the divide code model.


INTERFACES       

    FILE                 ROUTINE CALLED     

    CMutil.c             void cm_smooth_corner(); 


REFERENCED FILES

    Inputs from and outputs to ARGS structure.
                     

NON-STANDARD FEATURES

    NONE

===============================================================================*/

/*=== INCLUDE FILES ====================*/

#include <math.h>

                                      

/*=== CONSTANTS ========================*/




/*=== MACROS ===========================*/



  
/*=== LOCAL VARIABLES & TYPEDEFS =======*/                         


    
           
/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/




                   
/*==============================================================================

FUNCTION void cm_divide()

AUTHORS                      

     2 Oct 1991     Jeffrey P. Murray

MODIFICATIONS   

    NONE

SUMMARY

    This function implements the divide code model.

INTERFACES       

    FILE                 ROUTINE CALLED     

    CMutil.c             void cm_smooth_corner(); 

RETURNED VALUE
    
    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/


/*=== CM_DIVIDE ROUTINE ===*/


void cm_divide(ARGS)  

{
    double den_lower_limit;  /* denominator lower limit */
	double den_domain;       /* smoothing range for the lower limit */
	double threshold_upper;  /* value above which smoothing occurs */
	double threshold_lower;  /* value below which smoothing occurs */
    double numerator;        /* numerator input */
	double denominator;      /* denominator input */
	double limited_den;      /* denominator value if limiting is needed */   
	double den_partial;      /* partial of the output wrt denominator */
	double out_gain;         /* output gain */
    double num_gain;         /* numerator gain */
	double den_gain;         /* denominator gain */
    
    Mif_Complex_t ac_gain;



    /* Retrieve frequently used parameters... */

    den_lower_limit = PARAM(den_lower_limit);
    den_domain = PARAM(den_domain);
    out_gain = PARAM(out_gain);
    num_gain = PARAM(num_gain);
    den_gain = PARAM(den_gain);
                                                    

    if (PARAM(fraction) == MIF_TRUE)    /* Set domain to absolute value */
        den_domain = den_domain * den_lower_limit;

    threshold_upper = den_lower_limit +   /* Set Upper Threshold */
                         den_domain;

    threshold_lower = den_lower_limit -   /* Set Lower Threshold */
                         den_domain;

    numerator = (INPUT(num) + PARAM(num_offset)) * num_gain; 
    denominator = (INPUT(den) + PARAM(den_offset)) * den_gain; 

    if ((denominator < threshold_upper) && (denominator >= 0)) {  /* Need to limit den...*/

        if (denominator > threshold_lower)   /* Parabolic Region */
            cm_smooth_corner(denominator,den_lower_limit,
                        den_lower_limit,den_domain,0.0,1.0,
                        &limited_den,&den_partial);

        else {                            /* Hard-Limited Region */
            limited_den = den_lower_limit;
            den_partial = 0.0;        
        }
    }
	else
    if ((denominator > -threshold_upper) && (denominator < 0)) {  /* Need to limit den...*/
        if (denominator < -threshold_lower)   /* Parabolic Region */
            cm_smooth_corner(denominator,-den_lower_limit,
                        -den_lower_limit,den_domain,0.0,1.0,
                        &limited_den,&den_partial);

        else {                            /* Hard-Limited Region */
            limited_den = -den_lower_limit;
            den_partial = 0.0;        
        }
    }
    else {                         /* No limiting needed */
        limited_den = denominator;
        den_partial = 1.0;
    }

    if (ANALYSIS != MIF_AC) {

        OUTPUT(out) = PARAM(out_offset) + out_gain * 
                         ( numerator/limited_den );
        PARTIAL(out,num) = out_gain * num_gain / limited_den;
        PARTIAL(out,den) = -out_gain * numerator * den_gain *
                            den_partial / (limited_den * limited_den);

    }
    else {
        ac_gain.real = out_gain * num_gain / limited_den;
        ac_gain.imag= 0.0;
        AC_GAIN(out,num) = ac_gain;

        ac_gain.real = -out_gain * numerator * den_gain *
                            den_partial / (limited_den * limited_den);
        ac_gain.imag= 0.0;
        AC_GAIN(out,den) = ac_gain;
    }

}

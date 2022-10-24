/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE potentiometer/cfunc.mod

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503-405
               

AUTHORS                      

    19 June 1992     Jeffrey P. Murray


MODIFICATIONS   

    19 June 1992     Jeffrey P. Murray
    22 October 2022  Holger Vogt
                                   

SUMMARY

    This file contains the functional description of the potentiometer
    code model.


INTERFACES       

    FILE                 ROUTINE CALLED     

    CMmacros.h           cm_message_send();                   


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

FUNCTION cm_potentiometer()

AUTHORS                      

    19 June 1992     Jeffrey P. Murray

MODIFICATIONS   

    19 June 1992     Jeffrey P. Murray

SUMMARY

    This function implements the potentiometer code model.

INTERFACES       

    FILE                 ROUTINE CALLED     

    CMmacros.h           cm_message_send();                   

RETURNED VALUE
    
    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== CM_POTENTIOMETER ROUTINE ===*/


void cm_potentiometer (ARGS)
{
    double position;     /* position of wiper contact */
    double resistance;   /* total resistance */
    double r_lower;      /* resistance from r0 to wiper */
    double r_upper;      /* resistance from wiper to r1 */
    double vr0;          /* voltage at r0 */
    double vr1;          /* voltage at r1 */
    double vwiper;       /* voltage at wiper */



    Mif_Complex_t ac_gain;
                   
                       

    /* Retrieve frequently used parameters... */

    position = PARAM(position);

    /* guard against 0 or 1
    FIXME: checking the parameter limits is not yet implemented */
    if (position <= 0)
       position = 1e-9;
    else if (position >= 1)
       position = 0.999999999;

    resistance = PARAM(r);

    /* Retrieve input voltages... */
    vr0 = INPUT(r0);
    vwiper = INPUT(wiper);
    vr1 = INPUT(r1);


    if ( PARAM(log) == FALSE ) {   

        /* Linear Variation in resistance w.r.t. position */
        r_lower = position * resistance;
        r_upper = resistance - r_lower;

    }
    else {        
        
        /* Logarithmic Variation in resistance w.r.t. position */
        r_lower = resistance / 
                  pow(10.0,(position * PARAM(log_multiplier)));
        r_upper = resistance - r_lower;

    }





    /* Output DC & Transient Values  */

    if(ANALYSIS != MIF_AC) {               
        OUTPUT(r0) = (vr0 - vwiper) / r_lower;
        OUTPUT(r1) = (vr1 - vwiper) / r_upper;
        OUTPUT(wiper) = ((vwiper - vr0)/r_lower) + ((vwiper - vr1)/r_upper);

        PARTIAL(r0,r0) = 1.0 / r_lower;
        PARTIAL(r0,r1) = 0.0;
        PARTIAL(r0,wiper) = -1.0 / r_lower;

        PARTIAL(r1,r0) = 0.0;
        PARTIAL(r1,r1) = 1.0 / r_upper;
        PARTIAL(r1,wiper) = -1.0 / r_upper;

        PARTIAL(wiper,r0) = -1.0 / r_lower;
        PARTIAL(wiper,r1) = -1.0 / r_upper;
        PARTIAL(wiper,wiper) = (1.0/r_lower) + (1.0/r_upper);

    }
    else {                       

        /*   Output AC Gain Values      */

        ac_gain.imag= 0.0;              

        ac_gain.real = -1.0 / r_lower;
        AC_GAIN(r0,r0) = ac_gain;

        ac_gain.real = 0.0;             
        AC_GAIN(r0,r1) = ac_gain;

        ac_gain.real = 1.0 / r_lower;             
        AC_GAIN(r0,wiper) = ac_gain;

        ac_gain.real = 0.0;
        AC_GAIN(r1,r0) = ac_gain;

        ac_gain.real = -1.0 / r_upper;             
        AC_GAIN(r1,r1) = ac_gain;

        ac_gain.real = 1.0 / r_upper;             
        AC_GAIN(r1,wiper) = ac_gain;

        ac_gain.real = 1.0 / r_lower;
        AC_GAIN(wiper,r0) = ac_gain;

        ac_gain.real = 1.0 / r_upper;             
        AC_GAIN(wiper,r1) = ac_gain;

        ac_gain.real = -(1.0/r_lower) - (1.0/r_upper);             
        AC_GAIN(wiper,wiper) = ac_gain;

    }

}





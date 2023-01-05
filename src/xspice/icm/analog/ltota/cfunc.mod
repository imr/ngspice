/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE ltota/cfunc.mod

Public Domain

The ngspice team


AUTHORS

    1 Jan 2023     Holger Vogt


MODIFICATIONS



SUMMARY

    This file contains the model-specific routines used to
    functionally describe the ltota code model.


INTERFACES

    FILE                 ROUTINE CALLED

    CMutil.c             void cm_smooth_corner();
                         void cm_smooth_discontinuity();
                         void cm_climit_fcn()

REFERENCED FILES

    Inputs from and outputs to ARGS structure.


NON-STANDARD FEATURES

    NONE

===============================================================================*/

/*=== INCLUDE FILES ====================*/




/*=== CONSTANTS ========================*/




/*=== MACROS ===========================*/




/*=== LOCAL VARIABLES & TYPEDEFS =======*/




/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/





/*==============================================================================

FUNCTION void cm_ota()

AUTHORS

     1 Jan 2023    Holger Vogt

MODIFICATIONS


SUMMARY

    This function implements the ltota code model.

INTERFACES

    FILE                 ROUTINE CALLED

    CMutil.c             void cm_smooth_corner();
                         void cm_smooth_discontinuity();
                         void cm_climit_fcn()

RETURNED VALUE

    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES

    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== CM_OTA ROUTINE ===*/

void cm_ota(ARGS)  /* structure holding parms,
                                       inputs, outputs, etc.     */
{
// input parameters
    double ref, g, iout, isource, isink, ioffset, vhigh, vlow, rclamp, epsilon;
// noise parameters, not yet implemented
    double en, enk, in, ink, incm, incmk;

    double mult, curout, vout;

    Mif_Complex_t ac_gain;

    /* Retrieve frequently used parameters... */

    ref = PARAM(ref);
    g = PARAM(g);
    iout = PARAM(iout);
    isource = PARAM(isource);
    isink = PARAM(isink);
    ioffset = PARAM(ioffset);
    vhigh = PARAM(vhigh);
    vlow = PARAM(vlow);
    rclamp = PARAM(rclamp);
    epsilon = PARAM(epsilon);

    /* Test to see if in3 or in4 are connected or are not both 0.0 */
    /* if not, assign 1 to multiplier */
    /* else multiplier equals the difference */

   if ( PORT_NULL(in3) || PORT_NULL(in4) || (INPUT(in3) == 0.0 && INPUT(in4) == 0.0)) {
        mult = 1.0;
    }
    else {
        mult = INPUT(in3) - INPUT(in4);
    }
    /* output current */
    curout = (ref - INPUT(in1) + INPUT(in2)) * mult * g + ioffset;
    
    /* Retrieve frequently used inputs... */

    /* output voltage */
    vout = INPUT(out7);
    
    
    
    /* outputs without any limiting */
    if (ANALYSIS != MIF_AC) {     /* DC & Transient Analyses */
        OUTPUT(out7) = -curout;

        PARTIAL(out7,in1) = mult * g;
        PARTIAL(out7,in2) = -mult * g;
        PARTIAL(out7,in3) = (ref - INPUT(in1) + INPUT(in2)) * g;
        PARTIAL(out7,in4) = -1 * (ref - INPUT(in1) + INPUT(in2)) * g;

        PARTIAL(out7,out7) = 0;
    }


}






/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE gain/cfunc.mod

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
    functionally describe the gain code model.


INTERFACES       

    FILE                 ROUTINE CALLED     

    N/A                  N/A                     


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

FUNCTION void cm_gain()

AUTHORS                      

     2 Oct 1991     Jeffrey P. Murray

MODIFICATIONS   

    NONE

SUMMARY

    This function implements the gain code model.

INTERFACES       

    FILE                 ROUTINE CALLED     

    N/A                  N/A


RETURNED VALUE
    
    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/


/*=== CM_GAIN ROUTINE ===*/
                                                   

void cm_gain(ARGS)   /* structure holding parms, inputs, outputs, etc.     */
{
    Mif_Complex_t ac_gain;

    if(ANALYSIS != MIF_AC) {
        OUTPUT(out) = PARAM(out_offset) + PARAM(gain) * 
                         ( INPUT(in) + PARAM(in_offset));
        PARTIAL(out,in) = PARAM(gain);
    }
    else {
        ac_gain.real = PARAM(gain);
        ac_gain.imag= 0.0;
        AC_GAIN(out,in) = ac_gain;
    }

}

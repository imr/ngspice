/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE seegenerator/cfunc.mod

Public Domain

Universty Duisburg-Essen
Duisburg, Germany
Project Flowspace

AUTHORS                      

    19 May 2025 Holger Vogt


MODIFICATIONS   



SUMMARY

    This file contains the model-specific routines used to
    functionally describe the see (single event effects) generator code model.


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

FUNCTION void cm_seegen()

AUTHORS                      

    19 May 2025 Holger Vogt

SUMMARY

    This function implements the see generator code model.

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

/*=== CM_ILIMIT ROUTINE ===*/

void cm_seegen(ARGS)  /* structure holding parms, 
                                       inputs, outputs, etc.     */
{
    double talpha;
    double tbeta;
    double tdelay;
    double inull;
    double out;
    double tcurr = TIME;

    Mif_Complex_t ac_gain;



    /* Retrieve frequently used parameters... */

    talpha = PARAM(talpha);
    tbeta = PARAM(tbeta);
    tdelay = PARAM(tdelay);
    inull = PARAM(inull);

    out = inull * (exp(-(tcurr-tdelay)/talpha) - exp(-(tcurr-tdelay)/tbeta));

    OUTPUT(out) = out;
}




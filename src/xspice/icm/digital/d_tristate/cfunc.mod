/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE d_tristate/cfunc.mod

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503-405
               

AUTHORS                      

    18 Nov 1991     Jeffrey P. Murray


MODIFICATIONS   

    26 Nov 1991    Jeffrey P. Murray
                                   

SUMMARY

    This file contains the functional description of the d_tristate
    code model.


INTERFACES       

    FILE                 ROUTINE CALLED     

    CMevt.c              void *cm_event_alloc()
                         void *cm_event_get_ptr()
                         


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

FUNCTION cm_d_tristate()

AUTHORS                      

    18 Nov 1991     Jeffrey P. Murray

MODIFICATIONS   

    26 Nov 1991     Jeffrey P. Murray

SUMMARY

    This function implements the d_tristate code model.

INTERFACES       

    FILE                 ROUTINE CALLED     

    CMevt.c              void *cm_event_alloc()
                         void *cm_event_get_ptr()

RETURNED VALUE
    
    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== CM_D_TRISTATE ROUTINE ===*/

/************************************************
*      The following is a model for a simple    *
*   digital tristate for the ATESSE Version     *
*   2.0 system. Note that this version has      *
*   a single delay for both input and enable... *
*   a more realistic model is anticipated in    *
*   the not-so-distant future.                  *
*                                               *
*   Created 11/18/91              J.P,Murray    *
*   Last Modified 11/26/91                      *
************************************************/


void cm_d_tristate(ARGS) 
{
    int   enable;    /* holding variable for enable input */



    /* Retrieve input values and static variables */
    enable = INPUT_STATE(enable);

    OUTPUT_STATE(out) = INPUT_STATE(in);
    OUTPUT_DELAY(out) = PARAM(delay);


    /* define input loading... */
    LOAD(in) = PARAM(input_load);
    LOAD(enable) = PARAM(enable_load);




    if (ZERO == enable) {

        OUTPUT_STRENGTH(out) = HI_IMPEDANCE;

    }
    else 
    if (UNKNOWN == enable) {

        OUTPUT_STRENGTH(out) = UNDETERMINED;

    }
    else {
    
        OUTPUT_STRENGTH(out) = STRONG;

    }
}
 
    

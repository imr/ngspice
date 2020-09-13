/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE d_pullup/cfunc.mod

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503-405
               

AUTHORS                      

    19 Nov 1991     Jeffrey P. Murray


MODIFICATIONS   

    19 Nov 1991    Jeffrey P. Murray
                                   

SUMMARY

    This file contains the functional description of the d_pullup
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

FUNCTION cm_d_pullup()

AUTHORS                      

    19 Nov 1991     Jeffrey P. Murray

MODIFICATIONS   

    19 Nov 1991     Jeffrey P. Murray

SUMMARY

    This function implements the d_pullup code model.

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

/*=== CM_D_PULLUP ROUTINE ===*/

/************************************************
*      The following is the model for the       *
*   digital pullup resistor for the             *
*   ATESSE Version 2.0 system.                  *
*                                               *
*   Created 11/19/91              J.P,Murray    *
************************************************/


void cm_d_pullup(ARGS) 

{
    LOAD(out) = PARAM(load);
    OUTPUT_STATE(out) = ONE;
    OUTPUT_STRENGTH(out) = RESISTIVE;
}

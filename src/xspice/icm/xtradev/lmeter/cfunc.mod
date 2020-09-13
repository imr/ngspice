/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE lmeter/cfunc.mod

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503-405
               

AUTHORS                      

    30 Jul 1991     Bill Kuhn


MODIFICATIONS   

     2 Oct 1991    Jeffrey P. Murray
                                   

SUMMARY

    This file contains the model-specific routines used to
    functionally describe the lmeter code model.


INTERFACES       

    FILE                 ROUTINE CALLED     

    CMmeters.c           double cm_netlist_get_l()
                         


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

FUNCTION void cm_lmeter()

AUTHORS                      

    30 Jul 1991     Bill Kuhn


MODIFICATIONS   

     2 Oct 1991    Jeffrey P. Murray

SUMMARY

    This function implements the lmeter code model.

INTERFACES       

    FILE                 ROUTINE CALLED     

    CMmeters.c           double cm_netlist_get_l()


RETURNED VALUE
    
    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== CM_LMETER ROUTINE ===*/
                                                   

void cm_lmeter (ARGS)
{

    double      leq;

    if(INIT) {
        leq = cm_netlist_get_l();
        STATIC_VAR(l) = leq;
    }
    else
        leq = STATIC_VAR(l);

    OUTPUT(out) = PARAM(gain) * leq;
}





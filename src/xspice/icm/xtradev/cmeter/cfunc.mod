/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE cmeter/cfunc.mod

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503-405
               

AUTHORS                      

    30 July 1991     Bill Kuhn


MODIFICATIONS   

    26 Sept 1991    Jeffrey P. Murray
                                   

SUMMARY

    This file contains the functional description of the cmeter
    code model.


INTERFACES       

    FILE                 ROUTINE CALLED     

    CMmeters.c           double cm_netlist_get_c()
                         


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

FUNCTION cm_cmeter()

AUTHORS                      

    30 July 1991     Bill Kuhn

MODIFICATIONS   

    26 Sept 1991    Jeffrey P. Murray

SUMMARY

    This function implements the cmeter code model.

INTERFACES       

    FILE                 ROUTINE CALLED     

    CMmeters.c           double cm_netlist_get_c()

RETURNED VALUE
    
    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== CM_CMETER ROUTINE ===*/

void cm_cmeter (ARGS)
{

    double      ceq;    /* holding variable for read capacitance value */

    if(INIT) {
        ceq = cm_netlist_get_c();
        STATIC_VAR(c) = ceq;
    }
    else
        ceq = STATIC_VAR(c);

    OUTPUT(out) = PARAM(gain) * ceq;

}





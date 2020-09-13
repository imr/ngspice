/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE d_fdiv/cfunc.mod

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503-405
               

AUTHORS                      

    10 Jul 1991     Jeffrey P. Murray


MODIFICATIONS   

    22 Aug 1991    Jeffrey P. Murray
    30 Sep 1991    Jeffrey P. Murray
                                   

SUMMARY

    This file contains the functional description of the f_div
    (frequency divider) code model.


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

#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <string.h>

                                      

/*=== CONSTANTS ========================*/




/*=== MACROS ===========================*/



  
/*=== LOCAL VARIABLES & TYPEDEFS =======*/                         


    
           
/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/




                   
/*==============================================================================

FUNCTION cm_d_fdiv()

AUTHORS                      

    10 Jul 1991     Jeffrey P. Murray

MODIFICATIONS   

    22 Aug 1991    Jeffrey P. Murray
    30 Sep 1991    Jeffrey P. Murray

SUMMARY

    This function implements the d_fdiv code model.

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

/*=== CM_D_FDIV ROUTINE ===*/

/************************************************
*      The following is the model for the       *
*   digital frequency divider for the           *
*   ATESSE Version 2.0 system.                  *
*                                               *
*   Created 7/10/91               J.P.Murray    *
************************************************/


void cm_d_fdiv(ARGS) 

{
    int               div_factor;       /* division factor */
                        

    Digital_State_t     *freq_in,       /* freq_in clock value      */
                    *freq_in_old,       /* previous freq_in value   */
                       *freq_out,       /* current output for fdiv  */
                   *freq_out_old;       /* previous output for fdiv */                            

    int                   *count,       /* counter value    */
                      *count_old;       /* previous counter value    */



    /*** Setup required state variables ***/

    if(INIT) {  /* initial pass */ 

        /* allocate storage */
        cm_event_alloc(0,sizeof(Digital_State_t));
        cm_event_alloc(1,sizeof(Digital_State_t));
        cm_event_alloc(2,sizeof(int));

        /* declare load values */
        LOAD(freq_in) = PARAM(freq_in_load);

        /* retrieve storage for the outputs */
        freq_in = freq_in_old = (Digital_State_t *) cm_event_get_ptr(0,0);
        freq_out = freq_out_old = (Digital_State_t *) cm_event_get_ptr(1,0);
        count = count_old = (int *) cm_event_get_ptr(2,0);

    }
    else {      /* Retrieve previous values */
                                              
        /* retrieve storage for the outputs */
        freq_in = (Digital_State_t *) cm_event_get_ptr(0,0);
        freq_in_old = (Digital_State_t *) cm_event_get_ptr(0,1);
        freq_out = (Digital_State_t *) cm_event_get_ptr(1,0);
        freq_out_old = (Digital_State_t *) cm_event_get_ptr(1,1);
        count = (int *) cm_event_get_ptr(2,0);
        count_old = (int *) cm_event_get_ptr(2,1);

    }

                                                                

    /*** Output the strength of freq_out (always strong)... ***/
    OUTPUT_STRENGTH(freq_out) = STRONG;
              

    /** Retrieve parameters */
    div_factor = PARAM(div_factor);     
                   

    /******* Determine analysis type and output appropriate values *******/

    if (0.0 == TIME) {   /****** DC analysis...output w/o delays ******/
                                  

        /* read initial count value, normalize, and if it is out of 
           bounds, set to "zero" equivalent */
        *count = PARAM(i_count);
        if ( (div_factor <= *count) || (0 > *count) ) {
            *count = 0;
            OUTPUT_STATE(freq_out) = *freq_out = *freq_out_old = ZERO; 
        }

        if ( (0 < *count) && (*count <= PARAM(high_cycles)) ) {
            OUTPUT_STATE(freq_out) = *freq_out = *freq_out_old = ONE; 
        }                    

    }

    else {      /****** Transient Analysis ******/

        /*** load current input value... ***/
        *freq_in = INPUT_STATE(freq_in);
                                     

        /**** Test to see if the input has provided an edge... ****/
        if ( (*freq_in != *freq_in_old)&&(*freq_in == 1) ) { 

            /** An edge has been provided...revise count value **/
            *count = *count_old + 1;

            /* If new count value is equal to the div_factor+1 value,
               need to normalize count to "1", and raise output */
            if ( ((div_factor+1) == *count)||(1 == *count) ) {
                *count = 1;                                     
                OUTPUT_STATE(freq_out) = *freq_out = ONE;
                OUTPUT_DELAY(freq_out) = PARAM(rise_delay);
            }
            else {
                /* If new count value is equal to the 
                   high_cycles+1 value, drop the output to ZERO   */
                if ( ( PARAM(high_cycles)+1) == *count ) {
                    OUTPUT_STATE(freq_out) = *freq_out = ZERO;
                    OUTPUT_DELAY(freq_out) = PARAM(fall_delay);
                }
                else {
                    OUTPUT_CHANGED(freq_out) = FALSE;
                }
            }
        }
        else { /** Output does not change!! **/

            OUTPUT_CHANGED(freq_out) = FALSE;

        }
    }
} 

      




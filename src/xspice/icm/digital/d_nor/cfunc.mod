/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE d_nor/cfunc.mod

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503-405
               

AUTHORS                      

    18 Jun 1991     Jeffrey P. Murray


MODIFICATIONS   

    30 Sep 1991    Jeffrey P. Murray
                                   

SUMMARY

    This file contains the functional description of the d_nor
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

#include "ngspice/inertial.h"

/*=== CONSTANTS ========================*/


/*=== MACROS ===========================*/

  
/*=== LOCAL VARIABLES & TYPEDEFS =======*/                         
    
           
/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/


/*==============================================================================

FUNCTION cm_d_nor()

AUTHORS                      

    18 Jun 1991     Jeffrey P. Murray

MODIFICATIONS   

    30 Sep 1991     Jeffrey P. Murray

SUMMARY

    This function implements the d_nor code model.

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
                   

/*=== CM_D_NOR ROUTINE ===*/

/************************************************
*      The following is the model for the       *
*   digital NOR gate for the                    *
*   ATESSE Version 2.0 system.                  *
*                                               *
*   Created 6/18/91               J.P.Murray    *
************************************************/

void cm_d_nor(ARGS) 

{
    int                    i,   /* generic loop counter index */
	                size;   /* number of input & output ports */

    Digital_State_t      val,   /* Output value. */
                        *out,   /* temporary output for buffers */
                       input;   /* temp storage for input bits  */    


    /** Retrieve size value... **/
    size = PORT_SIZE(in);

    /*** Setup required state variables ***/

    if (INIT) {  /* initial pass */ 
        /* allocate storage for the outputs */

        cm_event_alloc(0, sizeof (Digital_State_t));

        /* Inertial delay? */

        STATIC_VAR(is_inertial) =
            cm_is_inertial(PARAM_NULL(inertial_delay) ? Not_set :
                         PARAM(inertial_delay));
        if (STATIC_VAR(is_inertial)) {
            /* Allocate storage for event time. */

            cm_event_alloc(1, sizeof (struct idata));
            ((struct idata *)cm_event_get_ptr(1, 0))->when = -1.0;
        }

        /* Prepare initial output. */

        out = (Digital_State_t *)cm_event_get_ptr(0, 0);
        *out = (Digital_State_t)(UNKNOWN + 1); // Force initial output.

        for (i=0; i<size; i++) LOAD(in[i]) = PARAM(input_load);
    } else {
        /* retrieve storage for the outputs */
        out = (Digital_State_t *) cm_event_get_ptr(0,0);
    }
                                     
    /*** Calculate new output value based on inputs ***/

    val = ONE;
    for (i=0; i<size; i++) {
        /* If a 1, set val low, and done. */

        if ( ONE == (input = INPUT_STATE(in[i])) ) {
            val = ZERO;
            break;
        } else {                    
            /* If an unknown input, set val to unknown. */

            if (UNKNOWN == input)
                val = UNKNOWN;
        }
    }       

    /*** Check for change and output appropriate values ***/

    if (val == *out) { /* output value is not changing */
        OUTPUT_CHANGED(out) = FALSE;
    } else {
        switch (val) {

            /* fall to zero value */
        case 0:
            OUTPUT_DELAY(out) = PARAM(fall_delay);
            break;
    
            /* rise to one value */
        case 1:
            OUTPUT_DELAY(out) = PARAM(rise_delay);
            break;
                                    
            /* unknown output */
        default:
            /* based on old value, add rise or fall delay */
            if (0 == *out) {  /* add rising delay */
                OUTPUT_DELAY(out) = PARAM(rise_delay);
            } else {                /* add falling delay */
                OUTPUT_DELAY(out) = PARAM(fall_delay);
            }
            break;
        }

        if (STATIC_VAR(is_inertial) && ANALYSIS == TRANSIENT) {
            struct idata *idp;

            idp = (struct idata *)cm_event_get_ptr(1, 0);
            if (idp->when <= TIME) {
                /* Normal transition. */

                idp->prev = *out;
                idp->when = TIME + OUTPUT_DELAY(out); // Actual output time
            } else if (val != idp->prev) {
                Digital_t ov = {idp->prev, STRONG};

                /* Third value: cancel earlier change and output as usual. */

                cm_schedule_output(1, 0, (idp->when - TIME) / 2.0, &ov);
		if (val == UNKNOWN) {
                    /* Delay based in idp->prev, not *out. */

                    if (idp->prev == ZERO)
                        OUTPUT_DELAY(out) = PARAM(rise_delay);
                    else
                        OUTPUT_DELAY(out) = PARAM(fall_delay);
                }
                idp->when = TIME + OUTPUT_DELAY(out); // Actual output time
            } else {
                /* Changing back: override pending change. */

                OUTPUT_DELAY(out) = (idp->when - TIME) / 2.0; // Override
		idp->when = -1.0;
            }
        }
        *out = val;
        OUTPUT_STATE(out) = val;
        OUTPUT_STRENGTH(out) = STRONG;
    }
}

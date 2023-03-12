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

#include "ngspice/inertial.h"

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
    Digital_t          *out;
    Digital_State_t     val, enable;
    Digital_Strength_t  str;
    struct idata       *idp;

    if (INIT) {  /* initial pass */ 
        /* define input loading... */
        LOAD(in) = PARAM(input_load);
        LOAD(enable) = PARAM(enable_load);
        OUTPUT_DELAY(out) = PARAM(delay);

        /* allocate storage for the previous output. */

        cm_event_alloc(0, sizeof (Digital_t));
        out = (Digital_t *)cm_event_get_ptr(0, 0);
        out->state = (Digital_State_t)(UNKNOWN + 1); // Force initial output.

        /* Inertial delay? */

        STATIC_VAR(is_inertial) =
            cm_is_inertial(PARAM_NULL(inertial_delay) ? Not_set :
                         PARAM(inertial_delay));
        if (STATIC_VAR(is_inertial)) {
            /* Allocate storage for event time. */

            cm_event_alloc(1, 2 * sizeof (struct idata));
            idp = (struct idata *)cm_event_get_ptr(1, 0);
            idp[1].when = idp[0].when = -1.0;
        }

        /* Prepare initial output. */

        out = (Digital_t *)cm_event_get_ptr(0, 0);
        out->state = (Digital_State_t)(UNKNOWN + 1); // Force initial output.
    } else {
        out = (Digital_t *)cm_event_get_ptr(0, 0);
    }

    /* Retrieve input values and static variables */

    val = INPUT_STATE(in);

    enable = INPUT_STATE(enable);
    if (ZERO == enable) {
        str = HI_IMPEDANCE;
    } else if (UNKNOWN == enable) {
        str = UNDETERMINED;
    } else {
        str = STRONG;
    }

    if (val == out->state && str == out->strength) {
        OUTPUT_CHANGED(out) = FALSE;
    } else {
        if (STATIC_VAR(is_inertial) && ANALYSIS == TRANSIENT) {
            int           d_cancel, s_cancel;

            idp = (struct idata *)cm_event_get_ptr(1, 0);
            d_cancel = (idp[0].when > TIME && val == idp[0].prev);
            s_cancel = (idp[1].when > TIME &&
                        str == (Digital_Strength_t)idp[1].prev);
            if ((d_cancel && s_cancel) ||
                (d_cancel && str == out->strength && TIME >= idp[1].when) ||
                (s_cancel && val == out->state && TIME >= idp[0].when)) {
                double when;

                /* Changing back: override pending change. */

                when =  d_cancel ? idp[0].when : idp[1].when;
                if (s_cancel && when > idp[1].when)
                    when = idp[1].when;

                OUTPUT_DELAY(out) = (when - TIME) / 2.0; // Override
                idp[1].when = idp[0].when = -1.0;
            } else {
                /* Normal transition, or third value during delay,
                 * or needs cancel followed by restore of
                 * the other component (fudge).
                 */

                OUTPUT_DELAY(out) = PARAM(delay);
                if (val != out->state) {
                    idp[0].prev = out->state;
                    idp[0].when = TIME + OUTPUT_DELAY(out);
                }
                if (str != out->strength) {
                    idp[1].prev = (Digital_State_t)out->strength;
                    idp[1].when = TIME + OUTPUT_DELAY(out);
                }
            }
        }
	out->state = val;
	out->strength = str;
	*(Digital_t *)OUTPUT(out) = *out;
    }
}

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

/* Extra state data for inertial delays. */

struct idata {
    double          when;
    Digital_State_t prev;
};

void cm_d_tristate(ARGS) 
{
    Digital_State_t *old, *new;
    int              enable;    /* holding variable for enable input */

    if (INIT) {  /* initial pass */ 

        /* define input loading... */
        LOAD(in) = PARAM(input_load);
        LOAD(enable) = PARAM(enable_load);

        /* allocate storage for the inputs. */

        cm_event_alloc(0, 2 * sizeof (Digital_State_t));
        if (PARAM(inertial_delay)) {
            struct idata *idp;

            /* Allocate storage for event time. */

            cm_event_alloc(1, 2 * sizeof (struct idata));
            idp = (struct idata *)cm_event_get_ptr(1, 0);
            idp[0].when = idp[1].when = -1.0;
        }
        new = old = (Digital_State_t *)cm_event_get_ptr(0, 0);
    } else {
        new = (Digital_State_t *)cm_event_get_ptr(0, 0);
        old = (Digital_State_t *)cm_event_get_ptr(0, 1);
    }

    /* Retrieve input values and static variables */

    new[0] = OUTPUT_STATE(out) = INPUT_STATE(in);
    OUTPUT_DELAY(out) = PARAM(delay);

    new[1] = enable = INPUT_STATE(enable);
    if (ZERO == enable) {
        OUTPUT_STRENGTH(out) = HI_IMPEDANCE;
    } else if (UNKNOWN == enable) {
        OUTPUT_STRENGTH(out) = UNDETERMINED;
    } else {
        OUTPUT_STRENGTH(out) = STRONG;
    }

    if (TIME == 0.0) {
        return;
    } else if (new[0] == old[0] && new[1] == old[1]) {
        OUTPUT_CHANGED(out) = FALSE;
    } else if (PARAM(inertial_delay)) {
        int           d_cancel, s_cancel;
        struct idata *idp;

        idp = (struct idata *)cm_event_get_ptr(1, 0);
        d_cancel = (idp[0].when > TIME && new[0] == idp[0].prev);
        s_cancel = (idp[1].when > TIME && new[1] == idp[1].prev);
        if ((d_cancel && s_cancel) ||
            (d_cancel && new[1] == old[1] && TIME >= idp[1].when) ||
            (s_cancel && new[0] == old[0] && TIME >= idp[0].when)) {
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

            if (new[0] != old[0]) {
                idp[0].prev = old[0];
                idp[0].when = TIME + OUTPUT_DELAY(out); // Actual output time
            }
            if (new[1] != old[1]) {
                idp[1].prev = old[1];
                idp[1].when = TIME + OUTPUT_DELAY(out); // Actual output time
            }
        }
    }
}

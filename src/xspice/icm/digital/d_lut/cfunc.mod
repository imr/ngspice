/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE d_lut/cfunc.mod

AUTHORS

    25 Aug 2016     Tim Edwards         efabless inc., San Jose, CA

SUMMARY

    This file contains the functional description of the d_lut
    code model.

LICENSE

    This software is in the public domain.

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

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <string.h>



/*=== CONSTANTS ========================*/




/*=== MACROS ===========================*/




/*=== LOCAL VARIABLES & TYPEDEFS =======*/




/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/



static void
lut_callback(ARGS, Mif_Callback_Reason_t reason)
{
    switch (reason) {
        case MIF_CB_DESTROY: {
            Digital_State_t *loc = STATIC_VAR (locdata);
	    if (loc) {
                free(loc);
                STATIC_VAR (locdata) = NULL;
            }
            break;
        }
    }
}


/*==============================================================================

FUNCTION cm_d_lut()

AUTHORS

    25 Aug 2016     Tim Edwards

SUMMARY

    This function implements the d_lut code model.

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

/*=== CM_D_LUT ROUTINE ===*/

/************************************************
*      The following is the model for the       *
*   digital n-input LUT gate                    *
*                                               *
*   Created 8/25/16               Tim Edwards   *
************************************************/


void cm_d_lut(ARGS)
{
    int         i,      /* generic loop counter index */
                j,      /* lookup index bit value */
                idx,    /* lookup index */
                size,   /* number of input & output ports */
                tablelen; /* length of table (2^size) */

    char        *table_string;

    Digital_State_t *out,      /* temporary output for buffers */
                    *out_old,  /* previous output for buffers  */
                    input,     /* temp storage for input bits  */
                    *lookup_table; /* lookup table */

    /** Retrieve size value and compute table length... **/
    size = PORT_SIZE(in);
    tablelen = 1 << size;

    /*** Setup required state variables ***/

    if (INIT) {  /* initial pass */

        /* allocate storage for the lookup table */
        STATIC_VAR (locdata) = calloc((size_t) tablelen, sizeof(Digital_State_t));
        lookup_table = STATIC_VAR (locdata);
	CALLBACK = lut_callback;

        /* allocate storage for the outputs */
        cm_event_alloc(0, sizeof(Digital_State_t));
        cm_event_alloc(1, size * (int) sizeof(Digital_State_t));

        /* set loading for inputs */
        for (i = 0; i < size; i++)
            LOAD(in[i]) = PARAM(input_load);

        /* retrieve storage for the outputs */
        out = out_old = (Digital_State_t *) cm_event_get_ptr(0, 0);

        /* read parameter string into lookup table */
        table_string = PARAM(table_values);
        for (idx = 0; idx < (int) strlen(table_string); idx++) {
            if (idx == tablelen)
                // If string is longer than 2^num_inputs, ignore
                // the extra values at the end
                break;
            if (table_string[idx] == '1')
                lookup_table[idx] = ONE;
            else if (table_string[idx] == '0')
                lookup_table[idx] = ZERO;
            else
                lookup_table[idx] = UNKNOWN;
        }
        // If string is shorter than 2^num_inputs, fill
        // the remainder of the lookup table with UNKNOWN values.
        for (; idx < tablelen; idx++)
            lookup_table[idx] = UNKNOWN;
    }
    else {      /* Retrieve previous values */

        /* retrieve lookup table */
        lookup_table = STATIC_VAR (locdata);

        /* retrieve storage for the outputs */
        out = (Digital_State_t *) cm_event_get_ptr(0, 0);
        out_old = (Digital_State_t *) cm_event_get_ptr(0, 1);
    }

    /*** Calculate new output value based on inputs and table ***/

    *out = ZERO;
    j = 1;
    idx = 0;
    for (i = 0; i < size; i++) {

        /* make sure this input isn't floating... */
        if (PORT_NULL(in) == FALSE) {

            /* use inputs to find index into lookup table */
            if ((input = INPUT_STATE(in[i])) == UNKNOWN) {
                *out = UNKNOWN;
                break;
            }
            else if (input == ONE) {
                idx += j;
            }
            j <<= 1;
        }
        else {
            /* at least one port is floating...output is unknown */
            *out = UNKNOWN;
            break;
        }
    }

    if (*out != UNKNOWN)
       *out = lookup_table[idx];

    /*** Determine analysis type and output appropriate values ***/

    if (ANALYSIS == DC) {   /** DC analysis...output w/o delays **/

        OUTPUT_STATE(out) = *out;

    }

    else {      /** Transient Analysis **/

        if (*out != *out_old) { /* output value is changing */

            switch (*out) {

            /* fall to zero value */
            case 0:
                OUTPUT_STATE(out) = ZERO;
                OUTPUT_DELAY(out) = PARAM(fall_delay);
                break;

            /* rise to one value */
            case 1:
                OUTPUT_STATE(out) = ONE;
                OUTPUT_DELAY(out) = PARAM(rise_delay);
                break;

            /* unknown output */
            default:
                OUTPUT_STATE(out) = *out = UNKNOWN;

                /* based on old value, add rise or fall delay */
                if (0 == *out_old)
                    OUTPUT_DELAY(out) = PARAM(rise_delay);
                else
                    OUTPUT_DELAY(out) = PARAM(fall_delay);
                break;
            }
        }
        else {                    /* output value not changing */
            OUTPUT_CHANGED(out) = FALSE;
        }
    }

    OUTPUT_STRENGTH(out) = STRONG;
}

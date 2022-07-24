/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE d_genlut/cfunc.mod

AUTHORS

    25 Aug 2016     Tim Edwards         efabless inc., San Jose, CA

SUMMARY

    This file contains the functional description of the d_genlut
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
genlut_callback(ARGS, Mif_Callback_Reason_t reason)
{
    switch (reason) {
        case MIF_CB_DESTROY: {
            Digital_t *loc = STATIC_VAR (locdata);
	    if (loc) {
                free(loc);
                STATIC_VAR (locdata) = NULL;
            }
            break;
        }
    }
}


/*==============================================================================

FUNCTION cm_d_genlut()

AUTHORS

    25 Aug 2016     Tim Edwards

SUMMARY

    This function implements the d_genlut code model.

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


void cm_d_genlut(ARGS)
{
    int i,          /* generic loop counter index */
        j,          /* lookup index bit value */
        k,          /* generic loop counter index */
        idx,        /* lookup index */
        ivalid,     /* check for valid input */
        isize,      /* number of input ports */
        osize,      /* number of output ports */
        dsize,      /* number of input delay params */
        rsize,      /* number of output rise delay params */
        fsize,      /* number of output fall delay params */
        lsize,      /* number of input load params */
        entrylen,   /* length of table per output (2^isize) */
        tablelen;   /* length of table (osize * (2^isize)) */

    char        *table_string;
    double      maxdelay,  /* maximum input-to-output delay */
        testdelay;

    Digital_State_t *in,       /* temp storage for input bits  */
        *in_old;   /* previous input for buffers  */
    Digital_t       *out,      /* temporary output for buffers */
        *out_old,  /* previous output for buffers  */
        *lookup_table; /* lookup table */

    /** Retrieve size values and compute table length... **/
    isize = PORT_SIZE(in);
    osize = PORT_SIZE(out);

    if (PARAM_NULL(input_load))
        lsize = 0;
    else
        lsize = PARAM_SIZE(input_load);

    if (PARAM_NULL(input_delay))
        dsize = 0;
    else
        dsize = PARAM_SIZE(input_delay);

    if (PARAM_NULL(rise_delay))
        rsize = 0;
    else
        rsize = PARAM_SIZE(rise_delay);

    if (PARAM_NULL(fall_delay))
        fsize = 0;
    else
        fsize = PARAM_SIZE(fall_delay);

    entrylen = (1 << isize);
    tablelen = osize * entrylen;

    /*** Setup required state variables ***/

    if (INIT) {  /* initial pass */

        /* allocate storage for the lookup table */
        STATIC_VAR (locdata) = calloc((size_t) tablelen, sizeof(Digital_t));
        lookup_table = STATIC_VAR (locdata);
	CALLBACK = genlut_callback;

        /* allocate storage for the outputs */
        cm_event_alloc(0, osize * (int) sizeof(Digital_t));
        cm_event_alloc(1, isize * (int) sizeof(Digital_State_t));

        /* set loading for inputs */
        for (i = 0; i < isize; i++)
            if (i < lsize)
                LOAD(in[i]) = PARAM(input_load[i]);
            else if (lsize > 0)
                LOAD(in[i]) = PARAM(input_load[lsize - 1]);
            else
                LOAD(in[i]) = 1.0e-12;

        /* retrieve storage for the outputs */
        out = out_old = (Digital_t *) cm_event_get_ptr(0, 0);
        in =  in_old = (Digital_State_t *) cm_event_get_ptr(1, 0);

        /* read parameter string into lookup table */
        table_string = PARAM(table_values);
        for (idx = 0; idx < (int)strlen(table_string); idx++) {
            if (idx == tablelen)
                // If string is longer than 2^num_inputs, ignore
                // the extra values at the end
                break;
            if (table_string[idx] == '1') {
                lookup_table[idx].state = ONE;
                lookup_table[idx].strength = STRONG;
            } else if (table_string[idx] == '0') {
                lookup_table[idx].state = ZERO;
                lookup_table[idx].strength = STRONG;
            } else if (table_string[idx] == 'z') {
                lookup_table[idx].state = UNKNOWN;
                lookup_table[idx].strength = HI_IMPEDANCE;
            } else {
                lookup_table[idx].state = UNKNOWN;
                lookup_table[idx].strength = UNDETERMINED;
            }
        }
        for (; idx < tablelen; idx++) {
            // If string is shorter than 2^num_inputs, fill
            // the remainder of the lookup table with UNKNOWN
            // values.
            lookup_table[idx].state = UNKNOWN;
            lookup_table[idx].strength = UNDETERMINED;
        }
    } else {    /* Retrieve previous values */

        /* retrieve lookup table */
        lookup_table = STATIC_VAR (locdata);

        /* retrieve storage for the inputs and outputs */
        out = (Digital_t *) cm_event_get_ptr(0, 0);
        out_old = (Digital_t *) cm_event_get_ptr(0, 1);
        in = (Digital_State_t *) cm_event_get_ptr(1, 0);
        in_old = (Digital_State_t *) cm_event_get_ptr(1, 1);
    }

    /*** Calculate new output value based on inputs and table ***/

    j = 1;
    idx = 0;
    ivalid = 1;
    for (k = 0; k < osize; k++) {
        out[k].state = UNKNOWN;
        out[k].strength = UNDETERMINED;
    }
    for (i = 0; i < isize; i++) {

        /* make sure this input isn't floating... */
        if (PORT_NULL(in) == FALSE) {

            /* use inputs to find index into lookup table */
            if ((in[i] = INPUT_STATE(in[i])) == UNKNOWN) {
                ivalid = 0;
                break;
            } else if (in[i] == ONE) {
                idx += j;
            }
            j <<= 1;
        } else {
            /* at least one port is floating...output is unknown */
            ivalid = 0;
            break;
        }
    }

    if (ivalid)
        for (k = 0; k < osize; k++)
            out[k] = lookup_table[idx + (k * entrylen)];

    /*** Determine analysis type and output appropriate values ***/

    if (ANALYSIS == DC) {   /** DC analysis...output w/o delays **/

        for (i = 0; i < osize; i++) {
            OUTPUT_STATE(out[i]) = out[i].state;
            OUTPUT_STRENGTH(out[i]) = out[i].strength;
        }
    }
    else {      /** Transient Analysis **/

        /* Determine maximum input-to-output delay */
        maxdelay = 0.0;
        for (i = 0; i < isize; i++)
            if (in[i] != in_old[i]) {
                if (i < dsize)
                    testdelay = PARAM(input_delay[i]);
                else if (dsize > 0)
                    testdelay = PARAM(input_delay[dsize - 1]);
                else
                    testdelay = 0.0;
                if (maxdelay < testdelay)
                    maxdelay = testdelay;
            }

        for (i = 0; i < osize; i++) {
            if (out[i].state != out_old[i].state) { /* output value is changing */

                OUTPUT_DELAY(out[i]) = maxdelay;
                switch (out[i].state) {

                    /* fall to zero value */
                case ZERO:
                    OUTPUT_STATE(out[i]) = ZERO;
                    if (i < fsize)
                        OUTPUT_DELAY(out[i]) += PARAM(fall_delay[i]);
                    else if (fsize > 0)
                        OUTPUT_DELAY(out[i]) += PARAM(fall_delay[fsize - 1]);
                    else
                        OUTPUT_DELAY(out[i]) += 1.0e-9;
                    OUTPUT_STRENGTH(out[i]) = out[i].strength;
                    break;

                    /* rise to one value */
                case ONE:
                    OUTPUT_STATE(out[i]) = ONE;
                    if (i < rsize)
                        OUTPUT_DELAY(out[i]) += PARAM(rise_delay[i]);
                    else if (rsize > 0)
                        OUTPUT_DELAY(out[i]) += PARAM(rise_delay[rsize - 1]);
                    else
                        OUTPUT_DELAY(out[i]) += 1.0e-9;
                    OUTPUT_STRENGTH(out[i]) = out[i].strength;
                    break;

                    /* unknown output */
                default:
                    OUTPUT_STATE(out[i]) = out[i].state = UNKNOWN;
                    OUTPUT_STRENGTH(out[i]) = out[i].strength;

                    /* based on old value, add rise or fall delay */
                    if (out_old[i].state == 0) {  /* add rising delay */
                        if (i < rsize)
                            OUTPUT_DELAY(out[i]) += PARAM(rise_delay[i]);
                        else if (rsize > 0)
                            OUTPUT_DELAY(out[i]) += PARAM(rise_delay[rsize - 1]);
                        else
                            OUTPUT_DELAY(out[i]) += 1.0e-9;
                    } else {              /* add falling delay */
                        if (i < fsize)
                            OUTPUT_DELAY(out[i]) += PARAM(fall_delay[i]);
                        else if (fsize > 0)
                            OUTPUT_DELAY(out[i]) += PARAM(fall_delay[fsize - 1]);
                        else
                            OUTPUT_DELAY(out[i]) += 1.0e-9;
                    }
                    break;
                }
            } else if (out[i].strength != out_old[i].strength) {
                /* output strength is changing */
                OUTPUT_STRENGTH(out[i]) = out[i].strength;
                switch (out[i].strength) {
                case STRONG:
                    if (out_old[i].state == 0) {        /* add falling delay */
                        if (i < fsize)
                            OUTPUT_DELAY(out[i]) += PARAM(fall_delay[i]);
                        else if (fsize > 0)
                            OUTPUT_DELAY(out[i]) += PARAM(fall_delay[fsize - 1]);
                        else
                            OUTPUT_DELAY(out[i]) += 1.0e-9;
                    } else {                    /* add rising delay */
                        if (i < rsize)
                            OUTPUT_DELAY(out[i]) += PARAM(rise_delay[i]);
                        else if (rsize > 0)
                            OUTPUT_DELAY(out[i]) += PARAM(rise_delay[rsize - 1]);
                        else
                            OUTPUT_DELAY(out[i]) += 1.0e-9;
                    }
                    break;
                default:
                    if (out_old[i].state == 0) {        /* add rising delay */
                        if (i < rsize)
                            OUTPUT_DELAY(out[i]) += PARAM(rise_delay[i]);
                        else if (rsize > 0)
                            OUTPUT_DELAY(out[i]) += PARAM(rise_delay[rsize - 1]);
                        else
                            OUTPUT_DELAY(out[i]) += 1.0e-9;
                    } else {                    /* add falling delay */
                        if (i < fsize)
                            OUTPUT_DELAY(out[i]) += PARAM(fall_delay[i]);
                        else if (fsize > 0)
                            OUTPUT_DELAY(out[i]) += PARAM(fall_delay[fsize - 1]);
                        else
                            OUTPUT_DELAY(out[i]) += 1.0e-9;
                    }
                    break;
                }
            } else {                  /* output value not changing */
                OUTPUT_CHANGED(out[i]) = FALSE;
            }
        }
    }
}

/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE delay/cfunc.mod

3-clause BSD

Copyright 2021
The ngspice team

AUTHORS

    12 June 2021     Holger Vogt


MODIFICATIONS



SUMMARY

    This file contains the functional description of the analog
    delay code model.


INTERFACES

    FILE                 ROUTINE CALLED

    CMutil.c             void cm_smooth_corner();

    CM.c                 void *cm_analog_alloc()
                         void *cm_analog_get_ptr()
                         int  cm_analog_integrate()
                              cm_delay_callback()



REFERENCED FILES

    Inputs from and outputs to ARGS structure.


NON-STANDARD FEATURES

    NONE

===============================================================================*/

/*=== INCLUDE FILES ====================*/

#include <stdlib.h>


/*=== CONSTANTS ========================*/




/*=== MACROS ===========================*/




/*=== LOCAL VARIABLES & TYPEDEFS =======*/

typedef struct {

    double   *buffer;       /* the storage array for input values   */
    int       buffer_size;  /* size of buffer */
    int       buff_write;   /* buffer write index */
    int       buff_del;     /* buffer index, delayed */
    int       step_count;   /* steps processed */
    double    tdelmin;      /* min delay, if controlled */
    double    tdelmax;      /* max delay, if controlled */
    double    tdelay;       /* time delay */
    double    tstep;        /* tran step size */
    double    tstop;        /* tran stop value */
    double    tprev;        /* previous time value */
    double    prev_val;     /* previous data value */
    double    start_val;    /* signal time 0 value */
} mLocal_Data_t;

/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/

static void cm_delay_callback(ARGS, Mif_Callback_Reason_t reason);

/*=== CM_DELAY ROUTINE ===*/

void cm_delay(ARGS)
{
    int buffer_size, delay_step;
    double delay, lcntrl;
    double delmin, delmax;

    mLocal_Data_t *loc;        /* Pointer to local static data, not to be included
                                       in the state vector */

    if (ANALYSIS != MIF_AC) {     /**** only Transient Analysis and dc ****/

        /** INIT: allocate storage **/

        if (INIT==1) {

            CALLBACK = cm_delay_callback;

            if (PARAM_NULL(buffer_size)) {
            /* size depends on TSTOP/TSTEP, if no parameter given */
                if (TSTEP > 0) /* called from tran analysis */
                    buffer_size = (int) (TSTOP/TSTEP) + 1;
                else /* called from op or dc analysis */
                    buffer_size = 1;
            }
            else {
                buffer_size = PARAM(buffer_size);
                if (buffer_size < 0) {
                    cm_message_send("Negative buffer size is not allowed "
                    "in a delay code model");
                    return;
                }
            }

            delay = PARAM(delay);
            if (delay < 0.0) {
                delay = 0.0;
                cm_message_send("Negative delay not allowed, set to 0");
            }

            /*** allocate static storage for *loc ***/
            if ((loc = (mLocal_Data_t *) (STATIC_VAR(locdata) = calloc(1,
                    sizeof(mLocal_Data_t)))) == (mLocal_Data_t *) NULL) {
                cm_message_send("Unable to allocate Local_Data_t "
                    "in cm_delay()");
                return;
            }
            /*** allocate static storage for the delay buffer ***/
            loc->buffer = (double *) calloc((size_t)buffer_size, sizeof(double));
            if (loc->buffer == (double *) NULL) {
                cm_message_send("Unable to allocate delay buffer "
                    "in cm_delay()");
                return;
            }
            loc->buffer_size = buffer_size;

            /* The delay is controlled by input delay_cnt */
            if (PARAM(has_delay_cnt) == MIF_TRUE) {
                if (PARAM_NULL(delmin))
                    loc->tdelmin = 0.0;
                else {
                    loc->tdelmin = PARAM(delmin);
                    if (loc->tdelmin < 0) {
                        loc->tdelmin = 0.0;
                        cm_message_send("Negative min delay not allowed, set to 0");
                    }
                    else if (loc->tdelmin > TSTOP) {
                        loc->tdelmin = TSTOP;
                        cm_message_send("min delay greater than final sim time not allowed, set to final time");
                    }
                }

                if (PARAM_NULL(delmax))
                    loc->tdelmax = TSTOP;
                else {
                    loc->tdelmax = PARAM(delmax);
                    if (loc->tdelmax < 0) {
                        loc->tdelmin = 0.0;
                        cm_message_send("Negative max delay not allowed, set to 0");
                    }
                    else if (loc->tdelmax > TSTOP) {
                        loc->tdelmax = TSTOP;
                        cm_message_send("max delay greater than final sim time not allowed, set to final time");
                    }
                }
                if (loc->tdelmax < loc->tdelmin) {
                    loc->tdelmax = loc->tdelmin;
                    cm_message_send("max delay smaller than min delay, set to min delay");
                }
            }

            loc->buff_write = 0;
            loc->buff_del = 0;
            loc->step_count = 0;
            loc->tdelay = delay;
            loc->tstop = TSTOP;
            loc->tstep = TSTEP;
            loc->tprev = 0.0;
            loc->prev_val = 0.0;
        }
        /* retrieve previous values */

        loc = STATIC_VAR (locdata);

        delay = loc->tdelay;
        if (delay == 0.0 && !PARAM(has_delay_cnt)) {
            OUTPUT(out) = INPUT(in);
            return;
        }

        if (delay < loc->tstep) {
            delay = 0.;
        }

        if (TIME == 0.0)
            loc->start_val = INPUT(in);

        /* input, simply interpolated to TSTEP        */
        double tacct = loc->step_count * loc->tstep;
        if (TIME >= tacct) {
            loc->buffer[loc->buff_write] = INPUT(in);
            /* next buffer location, circular, for writing */
            loc->buff_write = (loc->buff_write + 1) % loc->buffer_size;
            loc->step_count++;
            loc->tprev = tacct;
            loc->prev_val = INPUT(in);
        }

        delmin = loc->tdelmin;
        delmax = loc->tdelmax;

        if (PARAM(has_delay_cnt) == MIF_TRUE) {
            if (!PORT_NULL(cntrl)) {
                lcntrl =  INPUT(cntrl);
                if (lcntrl < 0)
                    lcntrl = 0.;
                else if (lcntrl > 1.)
                    lcntrl = 1.;
            }
            else {
                lcntrl = 0;
            }
            delay = (delmax - delmin) * lcntrl + delmin;
        }

        /* time not yet advanced for delay output */
        if (TIME < delay) {
            OUTPUT(out) = loc->start_val;
        }
        else if (delay == 0)
            OUTPUT(out) = INPUT(in);
        else
            OUTPUT(out) = loc->buffer[loc->buff_del];

        /* add offset to reduce error */
        /* FIXME: may not be needed if (better) interpolation */
        delay += 0.5 * loc->tstep;

        /* delay in steps */
        delay_step = (int)(delay / loc->tstep);
        /* FIXME: For whatever reason the model is assessed two times per time step */

        /* next readout location, delayed after writing */
        if ((loc->buff_write - delay_step) >= 0)
            loc->buff_del = loc->buff_write - delay_step;
        else
            loc->buff_del = loc->buff_write - delay_step + loc->buffer_size;
    }

    else {                    /**** all others ****/
        OUTPUT(out) = INPUT(in);
    }
}

/* free the memory created locally */
static void cm_delay_callback(ARGS, Mif_Callback_Reason_t reason)
{
    switch (reason) {
        case MIF_CB_DESTROY: {
            mLocal_Data_t *loc = (mLocal_Data_t *) STATIC_VAR(locdata);
            if (loc == (mLocal_Data_t *) NULL) {
                break;
            }

            if (loc->buffer != (double *) NULL) {
                free(loc->buffer);
            }

            free(loc);

            STATIC_VAR(locdata) = NULL;
            break;
        }
    }
} /* end of function cm_delay_callback */

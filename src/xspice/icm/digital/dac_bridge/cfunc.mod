/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE dac_bridge/cfunc.mod

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503-405
               

AUTHORS                      

    3 Jun 1991     Jeffrey P. Murray

MODIFICATIONS   

    16 Aug 1991    Jeffrey P. Murray
     2 Oct 1991    Jeffrey P. Murray

SUMMARY

    This file contains the model-specific routines used to
    functionally describe the dac_bridge code model.

INTERFACES       

    FILE                 ROUTINE CALLED     

    CM.c                 void *cm_analog_alloc()
                         void *cm_analog_get_ptr()
                         int cm_analog_set_perm_bkpt()

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

FUNCTION cm_dac_bridge()

AUTHORS                      

    3 Jun 1991     Jeffrey P. Murray

MODIFICATIONS   

    16 Aug 1991    Jeffrey P. Murray
     2 Oct 1991    Jeffrey P. Murray

SUMMARY

    This function implements the dac_bridge code model.

INTERFACES       

    FILE                 ROUTINE CALLED     

    CM.c                 void *cm_analog_alloc()
                         void *cm_analog_get_ptr()
                         int cm_analog_set_perm_bkpt()

    CMevt.c              void *cm_event_alloc()
                         void *cm_event_get_ptr()

RETURNED VALUE
    
    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

=============================================================================*/

/* Instances of this structure track digital input changes. */

struct d_data {
    Digital_State_t i;         // Input value.
    double          i_changed; // Time of input change.
};

/* Relative output value for multi-bit input. */

static double get_out_val(struct d_data *dp, int size)
{
    double v;
    int    i;

    for (i = size - 1, v = 0.0; i >= 0; --i) {
        v /= 2.0;
        switch (dp[i].i) {
        case ONE:
            v += 0.5;
            break;
        case UNKNOWN:
            v += 0.25;
            break;
        default:
            break;
        }
    }
    return v;
}

/*=== CM_DAC_BRIDGE ROUTINE ===*/

/************************************************
*      The following is the model for the       *
*   digital-to-analog nodebridge for the        *
*   ATESSE Version 2.0 system.                  *
*                                               *
*   Created 6/3/91                J.P.Murray    *
************************************************/

void cm_dac_bridge(ARGS) 
{
  double     out_low,       /* analog output value corresponding to '0'
                               digital input 	*/
            out_high,       /* analog output value corresponding to '1'
                               digital input 	*/
           out_undef,       /* analog output value corresponding to 'U'
                              digital input 	*/
              t_rise,       /* rise time...used to produce d(out)/d(time)
                               values for gradual change in analog output. */
              t_fall,       /* fall time...used to produce d(out)/d(time)
                               values for gradual change in analog output. */
                *out,       /* array holding all output values  */
            *out_old,       /* array holding previous output values */
           level_inc,       /* incremental level value out_high - out_low */
          rise_slope,       /* level_inc divided by t_rise */
          fall_slope,       /* level_inc divided by t_fall */
            time_inc;       /* time increment since last analog call */

   int             i,       /* generic loop counter index */
               multi,       /* Multi-bit in, single real out. */
             size_in,       /* number of input ports */
            size_out;       /* number of output ports */

   struct d_data *in,       /* base address of array holding all input
                               values  */
             *in_old;       /* array holding previous input values */


    /* Read in model parameters. **/
                              
    out_low = PARAM(out_low);
    out_high = PARAM(out_high);
    t_rise = PARAM(t_rise);
    t_fall = PARAM(t_fall);

    /* Test to see if out_low and out_high were specified, but */
    /* out_undef was not...                                    */
    /* if so, take out_undef as mean of out_high and out_low.  */

    if (!PARAM_NULL(out_low) && !PARAM_NULL(out_high) && 
         PARAM_NULL(out_undef)) {
       out_undef = out_low + (out_high - out_low) / 2.0;
    } else {
       out_undef = PARAM(out_undef);
    }                                 

    /* determine "width" of the node bridge... */

    size_in = PORT_SIZE(in);
    size_out = PORT_SIZE(out);
    multi = (size_in != size_out && size_out == 1);
    if (!multi) {
        if (size_in < size_out)
            size_out = size_in;
        else
            size_in = size_out;
    }


    if (INIT) {  /*** Test for INIT == TRUE. If so, allocate storage, etc. ***/
        if (size_in != size_out && size_out != 1) {
                cm_message_printf("Error: %d input ports with %d outputs",
                                  size_in, size_out);
        }

        /* Allocate storage for inputs */

        cm_event_alloc(0, size_in * (int)sizeof(struct d_data));
                      
        /* Allocate storage for outputs */

        cm_analog_alloc(0, size_out * (int)sizeof(double));
        
        /* Retrieve allocated addresses. */

        in = in_old = (struct d_data *) cm_event_get_ptr(0, 0);
        out = (double *) cm_analog_get_ptr(0, 0);

        /* read current input values */
        for (i = 0; i < size_in; i++) {
            in[i].i = INPUT_STATE(in[i]);
        }

        /* Output initial analog levels based on input values */

        if (multi) {
            /* Multi-bit input, single_output. */

            OUTPUT(out[0]) = *out =
                get_out_val(in, size_in) * (out_high - out_low) + out_low;
        } else {
            for (i = 0; i < size_in; i++) { /* assign addresses */
                switch (in[i].i) {
                case ZERO: out[i] = out_low;
                    break;

                case UNKNOWN: out[i] = out_undef;
                    break;

                case ONE: out[i] = out_high;
                    break;
                }
                OUTPUT(out[i]) = out[i];
            }
        }
        for (i = 0; i < size_in; i++)
            LOAD(in[i]) = PARAM(input_load);
        return;
    }

    /* This is not an initialization pass...read in parameters,
       retrieve storage addresses and calculate new outputs, if required.
    */

    /** Retrieve previous values... **/

    in = (struct d_data *) cm_event_get_ptr(0, 0);
    in_old= (struct d_data *) cm_event_get_ptr(0, 1);

    /* assign analog addresses */
    out = (double *) cm_analog_get_ptr(0, 0);
    out_old = (double *) cm_analog_get_ptr(0, 1);

    /* read current input values */
    for (i = 0; i < size_in; i++) {
        in[i].i = INPUT_STATE(in[i]);
    }

    switch (CALL_TYPE) {
        double          when, iota, vout, interval[2];
        int             step, step_count;

    case EVENT:  /** discrete call... **/
        /* Test to see if any change has occurred in an input */
        /* since the last digital call...                     */ 

        for (i = 0; i < size_in; i++) {
            if (in[i].i != in_old[i].i) { /* if there has been a change... */
                /* post current time as a breakpoint */

                cm_analog_set_perm_bkpt(TIME);

                if (multi) {
                    in[0].i_changed = TIME;
                    break;
                } else {
                    in[i].i_changed = TIME;
                }
            }
        }
        break;

    case ANALOG:    /** analog call... **/
        level_inc = out_high - out_low;
        rise_slope = level_inc / t_rise;
        fall_slope = level_inc / t_fall;

        time_inc = T(0) - T(1);

        if (multi) {
            double v, target;
            int    changed;

            /* Multi-bit input, single_output. */

            v = get_out_val(in, size_in);
            if (TIME == 0.0) {
                OUTPUT(out[0]) = *out = v * level_inc + out_low;;
                return;
            }
            vout = (out_old[0] - out_low) / level_inc; // Normalise.

            for (i = 0, changed = 0; i < size_in; i++) {
                if (in_old[i].i != in[i].i) {
                    changed = 1;
                    break;
                }
            }

            if (!changed) {
                if (vout < v) {
                    /* Continue rising. */

                    vout += time_inc / t_rise;
                    if (vout > v)
                        vout = v;
                } else {
                    /* Continue falling. */

                    vout -= time_inc / t_fall;
                    if (vout < v)
                        vout = v;
                }
            } else {
                /* There has been a change in input since the last
                   analog access. Determine when the change occurred
                   and calculate the current output, then set a breakpoint
                   for completion of the current transition.
                */

                iota = time_inc * 1e-7; // Ignorable
                if (T(0) - in[0].i_changed < iota) {
                    /* Previous input value in force for whole step. */

                    step_count = 1;
                    step = 0;
                    interval[0] = time_inc;
                } else if (in[0].i_changed - T(1) < iota) {
                    /* New input value in force for whole step.
                     * Includes common no-change case where new == old.
                     */

                    step_count = 2;
                    step = 1;
                    interval[1] = time_inc;
                } else {
                    /* Calculate both sides of change. */

                    step_count = 2;
                    step = 0;
                    interval[0] = in[0].i_changed - T(1);
                    interval[1] = T(0) - in[0].i_changed;
                }

                when = -1.0;
                for (; step < step_count; ++step) {
                    int last_step = (step == step_count - 1);

                    if (step == 0)
                        target = get_out_val(in_old, size_in);
                    else
                        target = v;

                    if (target > vout) {
                        /* Rising. */

                        vout += interval[step] / t_rise;
                        if (vout > v)
                            vout = v;
                        else if (last_step)
                            when = (v - vout) * t_rise;
                    } else if (target < vout) {
                        /* Falling. */

                        vout -= interval[step] / t_fall;
                        if (vout < v)
                            vout = v;
                        else if (last_step)
                            when = (vout - v) * t_fall;
                    }
                }
                if (when > 0.0)
                    cm_analog_set_perm_bkpt(when + TIME);
            }
            out[0] = vout * level_inc + out_low;
            OUTPUT(out[0]) = out[0];
            return;
        }

        /* Multiple single-bit conversions. */

        for (i = 0; i < size_in; i++) {
            if ( 0.0 == TIME ) {  /*** DC analysis ***/
                switch (in[i].i) {

                case ONE:
                    vout = out_high;
                    break;

                case ZERO:
                    vout = out_low;
                    break;

                case UNKNOWN:
                    vout = out_undef;
                    break;
                }
            } else if ( in_old[i].i == in[i].i ) {
                /*** Transient Analysis from here on. ***/

                /* There has been no change in
                   this digital input since the
                   last analog call...           */

                switch (in[i].i) {
                case ZERO:
                    if (out_old[i] > out_low) { /* output still dropping */
                        vout = out_old[i] - fall_slope * time_inc;
                        if (out_low > vout)
                            vout = out_low;
                    } else { /* output at out_low */
                        vout = out_low;
                    }
                    break;

                case ONE:
                    if (out_old[i] < out_high) { /* output still rising */
                        vout = out_old[i] + rise_slope * time_inc;
                        if (out_high < vout)
                            vout = out_high;
                    } else { /* output at out_high */
                        vout = out_high;
                    }
                    break;

                case UNKNOWN:
                    if (out_old[i] < out_undef) {     /* output still rising */
                        vout = out_old[i] + rise_slope * time_inc;
                        if (out_undef < vout)
                            vout = out_undef;
                    } else {
                        if (out_old[i] > out_undef) { /* output still falling */
                            vout = out_old[i] - fall_slope * time_inc;
                            if (out_undef > vout)
                                vout = out_undef;
                        } else {                     /* output at out_undef */
                            vout = out_undef;
                        }
                    }
                    break;
                }
            } else {
                /* There HAS been a change in this digital input
                   since the last analog access. Determine when the change
                   occurred and calculate the current output, then
                   set a breakpoint for completion of the current transition.
                */

                iota = (T(0) - T(1)) * 1e-7; // Ignorable
                if (T(0) - in[i].i_changed < iota) {
                    /* Previous input value in force for whole step. */

                    step_count = 1;
                    step = 0;
                    interval[0] = T(0) - T(1);
                } else if (in[i].i_changed - T(1) < iota) {
                    /* New input value in force for whole step.
                     * Includes common no-change case where new == old.
                     */

                    step_count = 2;
                    step = 1;
                    interval[1] = T(0) - T(1);
                } else {
                    /* Calculate both sides of change. */

                    step_count = 2;
                    step = 0;
                    interval[0] = in[i].i_changed - T(1);
                    interval[1] = T(0) - in[i].i_changed;
                }

                when = -1.0;
                vout = out_old[i];
                for (; step < step_count; ++step) {
                    Digital_State_t drive;
                    int             last_step = (step == step_count - 1);

                    if (step == 0)
                        drive = in_old[i].i;
                    else
                        drive = in[i].i;

                    switch (drive) {
                    case ZERO:
                        if (vout <= out_low)
                            break;
                        vout -= fall_slope * interval[step];
                        if (vout < out_low)
                            vout = out_low;
                        else if (last_step)
                            when = (vout - out_low) / fall_slope;
                        break;
                    case ONE:
                        if (vout >= out_high)
                            break;
                        vout += rise_slope * interval[step];
                        if (vout > out_high)
                            vout = out_high;
                        else if (last_step)
                            when = (out_high - vout) / rise_slope;
                        break;
                    case UNKNOWN:
                        if (vout > out_undef) {
                            vout -= fall_slope * interval[step];
                            if (vout < out_undef)
                                vout = out_undef;
                            else if (last_step)
                                when = (vout - out_undef) / fall_slope;
                        } else {
                            vout += rise_slope * interval[step];
                            if (vout > out_undef)
                                vout = out_undef;
                            else if (last_step)
                                when = (out_undef - vout) / rise_slope;
                        }
                        break;
                    }
                }
                if (when > 0.0)
                    cm_analog_set_perm_bkpt(when + TIME);
            }
            OUTPUT(out[i]) = out[i] = vout;
        }
        break;

    default:
        break;
    }
}

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

==============================================================================*/


/* Instances of this structure track digital input changes. */

struct d_data {
    Digital_State_t i;         // Input value.
    double          i_changed; // Time of input change.
};

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
	        size;       /* number of input & output ports */

   struct d_data  *in,      /* base address of array holding all input
                               values  */
              *in_old;       /* array holding previous input values */


    /* determine "width" of the node bridge... */

    size = PORT_SIZE(in);               

    /** Read in remaining model parameters **/
                              
    out_low = PARAM(out_low);
    out_high = PARAM(out_high);
    t_rise = PARAM(t_rise);
    t_fall = PARAM(t_fall);

    /* Test to see if out_low and out_high were specified, but */
    /* out_undef was not...                                    */
    /* if so, take out_undef as mean of out_high and out_low.  */

    if (!PARAM_NULL(out_low) && !PARAM_NULL(out_high) && 
         PARAM_NULL(out_undef) ) {
       out_undef = out_low + (out_high - out_low) / 2.0;
    } else {
       out_undef = PARAM(out_undef);
    }                                 

    if (INIT) {  /*** Test for INIT == TRUE. If so, allocate storage, etc. ***/
        /* Allocate storage for inputs */

        cm_event_alloc(0, size * (int) sizeof(struct d_data));
                      
        /* Allocate storage for outputs */

        cm_analog_alloc(0, size * (int) sizeof(double));
        
        /* Retrieve allocated addresses. */

        in = in_old = (struct d_data *) cm_event_get_ptr(0, 0);
        out = (double *) cm_analog_get_ptr(0, 0);

        /* read current input values */
        for (i=0; i<size; i++) {
            in[i].i = INPUT_STATE(in[i]);
        }

        /* Output initial analog levels based on input values */

        for (i=0; i<size; i++) { /* assign addresses */
            switch (in[i].i) {
                case ZERO: out[i] = out_low;
                        break;

                case UNKNOWN: out[i] = out_undef;
                        break;

                case ONE: out[i] = out_high;
                        break;
            }
            OUTPUT(out[i]) = out[i];
            LOAD(in[i]) = PARAM(input_load);
        }
        return;
    } else {    /*** This is not an initialization pass...read in parameters,
                   retrieve storage addresses and calculate new outputs,
                   if required. ***/

        /** Retrieve previous values... **/

        /* assign discrete addresses */

        in = (struct d_data *) cm_event_get_ptr(0, 0);
        in_old= (struct d_data *) cm_event_get_ptr(0, 1);

        /* assign analog addresses */
        out = (double *) cm_analog_get_ptr(0, 0);
        out_old = (double *) cm_analog_get_ptr(0, 1);

        /* read current input values */
        for (i=0; i<size; i++) {
            in[i].i = INPUT_STATE(in[i]);
        }
    }
    

    switch (CALL_TYPE) {

    case EVENT:  /** discrete call... **/
        /* Test to see if any change has occurred in an input */
        /* since the last digital call...                     */ 

        for (i=0; i<size; i++) {
            if (in[i].i != in_old[i].i) { /* if there has been a change... */
                in[i].i_changed = TIME;

                /* post current time as a breakpoint */

                cm_analog_set_perm_bkpt(TIME);
            }
        }
        break;

    case ANALOG:    /** analog call... **/

        level_inc = out_high - out_low;
        rise_slope = level_inc / t_rise;
        fall_slope = level_inc / t_fall;

        time_inc = TIME - T(1);

        for (i=0; i<size; i++) {
            if ( 0.0 == TIME ) {  /*** DC analysis ***/
                switch (in[i].i) {

                case ONE:
                    out[i] = out_high;
                    break;

                case ZERO:
                    out[i] = out_low;
                    break;

                case UNKNOWN:
                    out[i] = out_undef;
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
                        out[i] = out_old[i] - fall_slope * time_inc;
                        if ( out_low > out[i])
                            out[i] = out_low;
                    } else { /* output at out_low */
                        out[i] = out_low;
                    }
                    break;

                case ONE:
                    if (out_old[i] < out_high) { /* output still rising */
                        out[i] = out_old[i] + rise_slope * time_inc;
                        if ( out_high < out[i])
                            out[i] = out_high;
                    } else { /* output at out_high */
                        out[i] = out_high;
                    }
                    break;

                case UNKNOWN:
                    if (out_old[i] < out_undef) {     /* output still rising */
                        out[i] = out_old[i] + rise_slope * time_inc;
                        if ( out_undef < out[i])
                            out[i] = out_undef;
                    } else {
                        if (out_old[i] > out_undef) { /* output still falling */
                            out[i] = out_old[i] - fall_slope * time_inc;
                            if ( out_undef > out[i])
                                out[i] = out_undef;
                        } else {                     /* output at out_undef */
                            out[i] = out_undef;
                        }
                    }
                    break;
                }
            } else {
                double          when, iota, vout, interval[2];
                int             step, step_count;

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
                out[i] = vout;
            }
            OUTPUT(out[i]) = out[i];
        }
    }
}

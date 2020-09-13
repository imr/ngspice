/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE square/cfunc.mod

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503-405


AUTHORS

    12 Apr 1991     Harry Li


MODIFICATIONS

     2 Oct 1991    Jeffrey P. Murray


SUMMARY

    This file contains the model-specific routines used to
    functionally describe the square (controlled squarewave
    oscillator) code model.


INTERFACES

    FILE                 ROUTINE CALLED

    CMmacros.h           cm_message_send();

    CM.c                 void *cm_analog_alloc()
                         void *cm_analog_get_ptr()
                         int cm_analog_set_temp_bkpt()


REFERENCED FILES

    Inputs from and outputs to ARGS structure.


NON-STANDARD FEATURES

    NONE

===============================================================================*/

/*=== INCLUDE FILES ====================*/

#include <stdlib.h>


/*=== CONSTANTS ========================*/

char *square_allocation_error = "\n**** Error ****\nSQUARE: Error allocating square block storage \n";
char *square_limit_error = "\n**** Error ****\nSQUARE: Smoothing domain value too large \n";
char *square_freq_clamp = "\n**** WARNING  ****\nSQUARE: Frequency extrapolation limited to 1e-16 \n";
char *square_array_error = "\n**** Error ****\nSQUARE: Size of control array different than frequency array \n";

#define INT1 1
#define T1   2
#define T2   3
#define T3   4
#define T4   5


/*=== MACROS ===========================*/


/*=== LOCAL VARIABLES & TYPEDEFS =======*/

typedef struct {
    Boolean_t tran_init;        /* for initialization of phase1) */
} Local_Data_t;


/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/

/*==============================================================================

FUNCTION void cm_square()

AUTHORS

    12 Apr 1991     Harry Li

MODIFICATIONS

     2 Oct 1991    Jeffrey P. Murray

SUMMARY

    This function implements the square (controlled squarewave
    oscillator) code model.

INTERFACES

    FILE                 ROUTINE CALLED

    CMmacros.h           cm_message_send();

    CM.c                 void *cm_analog_alloc()
                         void *cm_analog_get_ptr()
                         int cm_analog_set_temp_bkpt()


RETURNED VALUE

    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES

    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== CM_SQUARE ROUTINE ===*/


/***********************************************************
*                                                          *
*  I                   <-dutycycle->                       *
*  I                                                       *
*  I                      out_high                         *
*  I                   t2    |     t3                      *
*  I                    \____v_____/                       *
*  I                    /          \                       *
*  I                                                       *
*  I                   /            \                      *
*  I                                                       *
*  I                  /              \                     *
*  I                                                       *
*  I-----------------I                I---------------     *
*         ^         t1               t4                    *
*         |                                                *
*      out_low                                             *
*                     t2 = t1 + t_rise                     *
*                     t4 = t3 + t_fall                     *
*                                                          *
***********************************************************/

void
cm_square(ARGS)
{
    int i;                      /* generic loop counter index */
    int cntl_size;              /* control array size         */
    int freq_size;              /* frequency array size       */
    int int_cycle;              /* integer number of cycles   */

    Mif_Value_t *x;             /* pointer to the control array values */
    Mif_Value_t *y;             /* pointer to the frequency array values */
    double cntl_input;          /* control input                       */
    double dout_din;            /* slope of the frequency array wrt the control
                                   array.  Used to extrapolate a frequency above
                                   and below the control input high and low level */
    double output_low;          /* output low */
    double output_hi;           /* output high */
    double dphase;              /* fractional part into cycle */
    double *phase;              /* pointer to the phase value */
    double *phase1;             /* pointer to the old phase value */
    double freq = 0.0;          /* frequency of the wave */
    double d_cycle;             /* duty cycle */
    double *t1;                 /* pointer containing the value of time1 */
    double *t2;                 /* pointer containing the value of time2 */
    double *t3;                 /* pointer containing the value of time3 */
    double *t4;                 /* pointer containing the value of time4 */
    double time1;               /* time1 = duty_cycle * period of the wave */
    double time2;               /* time2 = time1 + risetime */
    double time3;               /* time3 = current time+time to end of period*/
    double time4;               /* time4 = time3 + falltime */
    double t_rise;              /* risetime */
    double t_fall;              /* falltime */

    Mif_Complex_t ac_gain;

    /**** Retrieve frequently used parameters... ****/

    cntl_size = PARAM_SIZE(cntl_array);
    freq_size = PARAM_SIZE(freq_array);
    output_low = PARAM(out_low);
    output_hi = PARAM(out_high);
    d_cycle = PARAM(duty_cycle);
    t_rise = PARAM(rise_time);
    t_fall = PARAM(fall_time);

    /* check and make sure that the control array is the
       same size as the frequency array */

    if (cntl_size != freq_size) {
        cm_message_send(square_array_error);
        return;
    }

    /* First time throught allocate memory */
    if (INIT == 1) {
        cm_analog_alloc(INT1, sizeof(double));
        cm_analog_alloc(T1, sizeof(double));
        cm_analog_alloc(T2, sizeof(double));
        cm_analog_alloc(T3, sizeof(double));
        cm_analog_alloc(T4, sizeof(double));

        STATIC_VAR(tran_init) = MIF_FALSE;
    }

    x = (Mif_Value_t*) &PARAM(cntl_array[0]);
    y = (Mif_Value_t*) &PARAM(freq_array[0]);

    if (ANALYSIS == MIF_DC) {

        /* initialize time values */
        t1 = (double *) cm_analog_get_ptr(T1, 0);
        t2 = (double *) cm_analog_get_ptr(T2, 0);
        t3 = (double *) cm_analog_get_ptr(T3, 0);
        t4 = (double *) cm_analog_get_ptr(T4, 0);

        *t1 = -1;
        *t2 = -1;
        *t3 = -1;
        *t4 = -1;

        OUTPUT(out) = output_low;
        PARTIAL(out, cntl_in) = 0;

    } else if (ANALYSIS == MIF_TRAN) {

        /* Retrieve previous values */

        phase  = (double *) cm_analog_get_ptr(INT1, 0);
        phase1 = (double *) cm_analog_get_ptr(INT1, 1);

        t1 = (double *) cm_analog_get_ptr(T1, 1);
        t2 = (double *) cm_analog_get_ptr(T2, 1);
        t3 = (double *) cm_analog_get_ptr(T3, 1);
        t4 = (double *) cm_analog_get_ptr(T4, 1);

        time1 = *t1;
        time2 = *t2;
        time3 = *t3;
        time4 = *t4;

        if (STATIC_VAR(tran_init) == MIF_FALSE) {
            *phase1 = 0.0;
            STATIC_VAR(tran_init) = MIF_TRUE;
        }

        /* Retrieve cntl_input value. */
        cntl_input = INPUT(cntl_in);

        /* Determine segment boundaries within which cntl_input resides */
        /*** cntl_input below lowest cntl_voltage ***/
        if (cntl_input <= x[0].rvalue) {
            dout_din = (y[1].rvalue - y[0].rvalue) / (x[1].rvalue - x[0].rvalue);
            freq = y[0].rvalue + (cntl_input - x[0].rvalue) * dout_din;

            if (freq <= 0) {
                cm_message_send(square_freq_clamp);
                freq = 1e-16;
            }

        } else if (cntl_input >= x[cntl_size-1].rvalue) {
            /*** cntl_input above highest cntl_voltage ***/
            dout_din = (y[cntl_size-1].rvalue - y[cntl_size-2].rvalue) /
                (x[cntl_size-1].rvalue - x[cntl_size-2].rvalue);
            freq = y[cntl_size-1].rvalue + (cntl_input - x[cntl_size-1].rvalue) * dout_din;

        } else {
            /*** cntl_input within bounds of end midpoints...
                 must determine position progressively & then
                 calculate required output. ***/

            for (i = 0; i < cntl_size - 1; i++) {

                if ((cntl_input < x[i+1].rvalue) && (cntl_input >= x[i].rvalue)) {

                    /* Interpolate to the correct frequency value */

                    freq = ((cntl_input - x[i].rvalue)/(x[i+1].rvalue - x[i].rvalue)) *
                        (y[i+1].rvalue-y[i].rvalue) + y[i].rvalue;
                }

            }

        }

        /* calculate the instantaneous phase */
        *phase = *phase1 + freq*(TIME - T(1));

        /* convert the phase to an integer */
        int_cycle = (int) *phase1;

        /* dphase is the percent into the cycle for
           the period */
        dphase = *phase1 - int_cycle;

        /* Calculate the time variables and the output value
           for this iteration */

        if ((time1 <= TIME) && (TIME <= time2)) {

            time3 = T(1) + (1 - dphase) / freq;
            time4 = time3 + t_fall;

            if (TIME < time2)
                cm_analog_set_temp_bkpt(time2);

            cm_analog_set_temp_bkpt(time3);
            cm_analog_set_temp_bkpt(time4);

            OUTPUT(out) = output_low + ((TIME - time1) / (time2 - time1)) *
                (output_hi - output_low);

        } else if ((time2 <= TIME) && (TIME <= time3)) {

            time3 = T(1) + (1.0 - dphase) / freq;
            time4 = time3 + t_fall;

            if (TIME < time3)
                cm_analog_set_temp_bkpt(time3);

            cm_analog_set_temp_bkpt(time4);

            OUTPUT(out) = output_hi;

        } else if ((time3 <= TIME) && (TIME <= time4)) {

            if (dphase > 1 - d_cycle)
                dphase = dphase - 1.0;

            /* subtract d_cycle from 1 because my initial definition
               of duty cyle was that part of the cycle which the output
               is low.  The more standard definition is the part of the
               cycle where the output is high. */
            time1 = T(1) + ((1-d_cycle) - dphase) / freq;
            time2 = time1 + t_rise;

            if (TIME < time4)
                cm_analog_set_temp_bkpt(time4);

            cm_analog_set_temp_bkpt(time1);
            cm_analog_set_temp_bkpt(time2);

            OUTPUT(out) = output_hi + ((TIME - time3) / (time4 - time3)) *
                (output_low - output_hi);

        } else {

            if (dphase > 1 - d_cycle)
                dphase = dphase - 1.0;

            /* subtract d_cycle from 1 because my initial definition
               of duty cyle was that part of the cycle which the output
               is low.  The more standard definition is the part of the
               cycle where the output is high. */
            time1 = T(1) + ((1-d_cycle) - dphase) / freq;
            time2 = time1 + t_rise;

            if ((TIME < time1) || (T(1) == 0))
                cm_analog_set_temp_bkpt(time1);

            cm_analog_set_temp_bkpt(time2);

            OUTPUT(out) = output_low;

        }

        PARTIAL(out, cntl_in) = 0.0;

        /* set the time values for storage */

        t1 = (double *) cm_analog_get_ptr(T1, 0);
        t2 = (double *) cm_analog_get_ptr(T2, 0);
        t3 = (double *) cm_analog_get_ptr(T3, 0);
        t4 = (double *) cm_analog_get_ptr(T4, 0);

        *t1 = time1;
        *t2 = time2;
        *t3 = time3;
        *t4 = time4;

    } else {                      /* Output AC Gain */

        /* This model has no AC capabilities */

        ac_gain.real = 0.0;
        ac_gain.imag = 0.0;
        AC_GAIN(out, cntl_in) = ac_gain;
    }
}

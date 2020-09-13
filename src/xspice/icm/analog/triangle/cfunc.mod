/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE triangle/cfunc.mod

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503-405

AUTHORS

    12 Apr 1991     Harry Li


MODIFICATIONS

     2 Oct 1991    Jeffrey P. Murray
     9 Sep 2012    Holger Vogt


SUMMARY

    This file contains the model-specific routines used to
    functionally describe the triangle (controlled trianglewave
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

char *triangle_allocation_error = "\n**** Error ****\nTRIANGLE: Error allocating triangle block storage \n";
char *triangle_freq_clamp = "\n**** Warning ****\nTRIANGLE: Extrapolated Minimum Frequency Set to 1e-16 Hz \n";
char *triangle_array_error = "\n**** Error ****\nTRIANGLE: Size of control array different than frequency array \n";

#define INT1 1
#define T1   2
#define T2   3
#define T3   4


/*=== MACROS ===========================*/


/*=== LOCAL VARIABLES & TYPEDEFS =======*/


/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/

/*==============================================================================

FUNCTION void cm_triangle()

AUTHORS

    12 Apr 1991     Harry Li

MODIFICATIONS

     2 Oct 1991    Jeffrey P. Murray
     9 Sep 2012    Holger Vogt

SUMMARY

    This function implements the triangle (controlled trianglewave
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

/*=== CM_TRIANGLE ROUTINE ===*/

/*****************************************************
*                                                    *
*  I         /\ <- output_high                       *
*  I        /  \                                     *
*  I       /    \                                    *
*  I      /      \                                   *
*  I     /        \                                  *
*  I    /          \                                 *
*  I   /            \                                 *
*  I  /              \                               *
*  I /                \                              *
*  I/------------------------------------------      *
*                       \              /             *
*                        \            /              *
*                         \          /               *
*                          \        /                *
*                           \      /                 *
*                            \    /                  *
*                             \  /                   *
*                              \/ <- output_low      *
*                                                    *
*****************************************************/

void
cm_triangle(ARGS)
{
    int i;                 /* generic loop counter index                      */
    int cntl_size;         /* size of the control array                       */
    int freq_size;         /* size of the frequency array                     */
    int int_cycle;         /* the number of cycles rounded to the nearest int */

    Mif_Value_t *x;        /* pointer holds the values of the control array */
    Mif_Value_t *y;        /* pointer holds the values of the freq array    */
    double cntl_input;     /* control input                                 */

    double dout_din;       /* partial out wrt to control input              */
    double output_low;     /* lowest point of the wave                      */
    double output_hi;      /* highest point of the wave                     */
    double dphase;         /* percent into the current phase of the cycle   */
    double *phase;         /* instantaneous phase value                     */
    double *phase1;        /* pointer to the previous phase value           */
    double freq = 0.0;     /* actual frequency of the wave                  */
    double d_cycle;        /* duty cycle                                    */
    double *t1;            /* pointer which stores time1                    */
    double *t2;            /* pointer which stores time2                    */
    double *t_end;         /* pointer which stores t_start                  */
    double time1;          /* time of high peak                             */
    double time2;          /* time of low peak                              */
    double t_start;        /* time of the beginning of each cycle           */

    Mif_Complex_t ac_gain;

    /**** Retrieve frequently used parameters... ****/


    cntl_size = PARAM_SIZE(cntl_array);
    freq_size = PARAM_SIZE(freq_array);
    output_low = PARAM(out_low);
    output_hi = PARAM(out_high);
    d_cycle = PARAM(duty_cycle);

    if(cntl_size != freq_size) {
        cm_message_send(triangle_array_error);
        return;
    }

    /* Allocate memory */

    if (INIT == 1) {
        cm_analog_alloc(INT1, sizeof(double));
        cm_analog_alloc(T1, sizeof(double));
        cm_analog_alloc(T2, sizeof(double));
        cm_analog_alloc(T3, sizeof(double));

        STATIC_VAR(tran_init) = MIF_FALSE;
    }

    x = (Mif_Value_t*) &PARAM(cntl_array[0]);
    y = (Mif_Value_t*) &PARAM(freq_array[0]);

    if (ANALYSIS == MIF_DC) {

        /* initialize time values */

        t1    = (double *) cm_analog_get_ptr(T1, 0);
        t2    = (double *) cm_analog_get_ptr(T2, 0);
        t_end = (double *) cm_analog_get_ptr(T3, 0);

        *t1    = -1;
        *t2    = -1;
        *t_end =  0;

        OUTPUT(out) = output_low;
        PARTIAL(out, cntl_in) = 0;

    } else if (ANALYSIS == MIF_TRAN) {

        /* Retrieve previous values and set equal to corresponding variables */

        phase  = (double *) cm_analog_get_ptr(INT1, 0);
        phase1 = (double *) cm_analog_get_ptr(INT1, 1);

        t1    = (double *) cm_analog_get_ptr(T1, 1);
        t2    = (double *) cm_analog_get_ptr(T2, 1);
        t_end = (double *) cm_analog_get_ptr(T3, 1);

        time1   = *t1;
        time2   = *t2;
        t_start = *t_end;

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
                cm_message_send(triangle_freq_clamp);
                freq = 1e-16;
            }

        } else if (cntl_input >= x[cntl_size-1].rvalue) {
            /*** cntl_input above highest cntl_voltage ***/
            dout_din = (y[cntl_size-1].rvalue - y[cntl_size-2].rvalue) /
                (x[cntl_size-1].rvalue - x[cntl_size-2].rvalue);
            freq = y[cntl_size-1].rvalue + (cntl_input - x[cntl_size-1].rvalue) * dout_din;
            /* freq = y[cntl_size-1].rvalue; */

        } else {
            /*** cntl_input within bounds of end midpoints...
                 must determine position progressively & then
                 calculate required output. ***/

            for (i = 0; i < cntl_size - 1; i++) {

                if ((cntl_input < x[i+1].rvalue) && (cntl_input >= x[i].rvalue)) {

                    /* Interpolate to the correct frequency value */

                    freq = ((cntl_input - x[i].rvalue) / (x[i+1].rvalue - x[i].rvalue)) *
                        (y[i+1].rvalue - y[i].rvalue) + y[i].rvalue;
                }
            }
        }

        /* Instantaneous phase is the old phase + frequency/(delta time)
           int_cycle is the integer value for the number cycles. */

        *phase = *phase1 + freq * (TIME - T(1));
        int_cycle = (int) *phase1;
        dphase = *phase1 - int_cycle;

        /* if the current time is greater than time1, but less than time2,
           calculate time2 and set the temporary breakpoint.  */
        if ((time1 <= TIME) && (TIME <= time2)) {

            time2 = T(1) + (1 - dphase) / freq;

            if (TIME < time2)
                cm_analog_set_temp_bkpt(time2);

            /* store the time that the next cycle is scheduled to begin */
            t_start = time2;

            /* set output value */
            OUTPUT(out) = output_hi - ((TIME - time1) / (time2 - time1)) *
                (output_hi - output_low);

        } else {

            /* otherwise, calculate time1 and time2 and set their respective
               breakpoints */

            if (dphase > d_cycle)
                dphase = dphase - 1.0;

            time1 = T(1) + (d_cycle - dphase) / freq;
            time2 = T(1) + (1 - dphase) / freq;

            if ((TIME < time1) || (T(1) == 0))
                cm_analog_set_temp_bkpt(time1);

            cm_analog_set_temp_bkpt(time2);

            /* set output value */
            OUTPUT(out) = output_low + ((TIME - t_start) / (time1 - t_start)) *
                (output_hi - output_low);
        }

        PARTIAL(out, cntl_in) = 0.0;

        /* set the time values for storage */

        t1    = (double *) cm_analog_get_ptr(T1, 0);
        t2    = (double *) cm_analog_get_ptr(T2, 0);
        t_end = (double *) cm_analog_get_ptr(T3, 0);

        *t1 = time1;
        *t2 = time2;
        *t_end = t_start;

    } else {                      /* Output AC Gain */

        /* This model has no AC capabilities */

        ac_gain.real = 0.0;
        ac_gain.imag = 0.0;
        AC_GAIN(out, cntl_in) = ac_gain;
    }
}

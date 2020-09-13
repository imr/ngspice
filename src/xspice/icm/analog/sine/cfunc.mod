/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE sine/cfunc.mod

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503-405


AUTHORS

    20 Mar 1991     Harry Li


MODIFICATIONS

     2 Oct 1991    Jeffrey P. Murray
     7 Sep 2012    Holger Vogt


SUMMARY

    This file contains the model-specific routines used to
    functionally describe the sine (controlled sine-wave
    oscillator) code model.


INTERFACES

    FILE                 ROUTINE CALLED

    CMmacros.h           cm_message_send();

    CM.c                 void *cm_analog_alloc()
                         void *cm_analog_get_ptr()


REFERENCED FILES

    Inputs from and outputs to ARGS structure.


NON-STANDARD FEATURES

    NONE

===============================================================================*/

/*=== INCLUDE FILES ====================*/

#include <math.h>
#include <stdlib.h>


/*=== CONSTANTS ========================*/

#define INT1 1

char *allocation_error = "\n**** Error ****\nSINE: Error allocating sine block storage \n";
char *limit_error = "\n**** Error ****\nSINE: Smoothing domain value too large \n";
char *sine_freq_clamp = "\n**** Warning ****\nSINE: Extrapolated frequency limited to 1e-16 Hz \n";
char *array_error = "\n**** Error ****\nSINE: Size of control array different than frequency array \n";


/*=== MACROS ===========================*/


/*=== LOCAL VARIABLES & TYPEDEFS =======*/


/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/


/*==============================================================================

FUNCTION void cm_sine()

AUTHORS

    20 Mar 1991     Harry Li

MODIFICATIONS

     2 Oct 1991    Jeffrey P. Murray
     7 Sep 2012    Holger Vogt

SUMMARY

    This function implements the sine (controlled sinewave
    oscillator) code model.

INTERFACES

    FILE                 ROUTINE CALLED

    CMmacros.h           cm_message_send();

    CM.c                 void *cm_analog_alloc()
                         void *cm_analog_get_ptr()

RETURNED VALUE

    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES

    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== CM_SINE ROUTINE ===*/

void
cm_sine(ARGS)
{
    int i;             /* generic loop counter index                    */
    int cntl_size;     /* control array size                            */
    int freq_size;     /* frequency array size                          */

    Mif_Value_t *x;    /* pointer to the control array values           */
    Mif_Value_t *y;    /* pointer to the frequency array values         */
    double cntl_input; /* control input                                 */
    double dout_din;   /* partial derivative of output wrt control in   */
    double output_low; /* output low value                              */
    double output_hi;  /* output high value                             */
    double *phase;     /* pointer to the instantaneous phase value      */
    double *phase1;    /* pointer to the previous value for the phase   */
    double freq=0.0;   /* frequency of the sine wave                    */
    double center;     /* dc offset for the sine wave                   */
    double peak;       /* peak voltage value for the wave               */
    double radian;     /* phase value in radians                        */

    Mif_Complex_t ac_gain;

    /**** Retrieve frequently used parameters... ****/

    cntl_size  = PARAM_SIZE(cntl_array);
    freq_size  = PARAM_SIZE(freq_array);
    output_low = PARAM(out_low);
    output_hi  = PARAM(out_high);

    if (cntl_size != freq_size) {
        cm_message_send(array_error);
        return;
    }

    if (INIT == 1) {

        cm_analog_alloc(INT1, sizeof(double));

    }

    x = (Mif_Value_t*) &PARAM(cntl_array[0]);
    y = (Mif_Value_t*) &PARAM(freq_array[0]);

    if (ANALYSIS == MIF_DC) {

        OUTPUT(out) = (output_hi + output_low) / 2;
        PARTIAL(out, cntl_in) = 0;
        phase = (double *) cm_analog_get_ptr(INT1, 0);
        *phase = 0;

    } else if (ANALYSIS == MIF_TRAN) {

        phase  = (double *) cm_analog_get_ptr(INT1, 0);
        phase1 = (double *) cm_analog_get_ptr(INT1, 1);

        /* Retrieve cntl_input value. */
        cntl_input = INPUT(cntl_in);

        /* Determine segment boundaries within which cntl_input resides */
        if (cntl_input <= x[0].rvalue) {

            /*** cntl_input below lowest cntl_voltage ***/
            dout_din = (y[1].rvalue - y[0].rvalue) / (x[1].rvalue - x[0].rvalue);
            freq = y[0].rvalue + (cntl_input - x[0].rvalue) * dout_din;

            if (freq <= 0) {
                cm_message_send(sine_freq_clamp);
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

            for (i = 0; i < cntl_size - 1; i++)
                if ((cntl_input < x[i+1].rvalue) && (cntl_input >= x[i].rvalue)) {
                    /* Interpolate to the correct frequency value */
                    freq = ((cntl_input - x[i].rvalue) / (x[i+1].rvalue - x[i].rvalue)) *
                        (y[i+1].rvalue - y[i].rvalue) + y[i].rvalue;
                }
        }

        /* calculate the peak value of the wave, the center of the wave, the
           instantaneous phase and the radian value of the phase */
        peak   = (output_hi - output_low) / 2;
        center = (output_hi + output_low) / 2;

        *phase = *phase1 + freq*(TIME - T(1));
        radian = *phase * 2.0 * M_PI;

        OUTPUT(out) = peak * sin(radian) + center;
        PARTIAL(out, cntl_in) = 0;

    } else {                /* Output AC Gain */

        ac_gain.real = 0.0;
        ac_gain.imag = 0.0;
        AC_GAIN(out, cntl_in) = ac_gain;

    }
}

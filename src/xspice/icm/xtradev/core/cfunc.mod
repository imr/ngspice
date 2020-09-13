/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE core/cfunc.mod

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503-405


AUTHORS

    24 Apr 1991     Jeffrey P. Murray


MODIFICATIONS

    24 Apr 1991    Jeffrey P. Murray
    26 Sep 1991    Jeffrey P. Murray


SUMMARY

    This file contains the functional description of the core
    code model.


INTERFACES

    FILE                 ROUTINE CALLED

    CMmacros.h           cm_message_send();

    CMutil.c             void cm_smooth_corner();


REFERENCED FILES

    Inputs from and outputs to ARGS structure.


NON-STANDARD FEATURES

    NONE

===============================================================================*/

/*=== INCLUDE FILES ====================*/

#include <stdlib.h>


/*=== CONSTANTS ========================*/

#define HYST 1
#define X_RISING TRUE
#define X_FALLING FALSE
#define PWL 1
#define HYSTERESIS 2


/*=== MACROS ===========================*/


/*=== LOCAL VARIABLES & TYPEDEFS =======*/


/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/


/*==============================================================================

FUNCTION cm_core()

AUTHORS

    24 Apr 1991     Jeffrey P. Murray

MODIFICATIONS

    24 Apr 1991    Jeffrey P. Murray
    26 Sep 1991    Jeffrey P. Murray

SUMMARY

    This function implements the core code model.

INTERFACES

    FILE                 ROUTINE CALLED

    CMmacros.h           cm_message_send();

    CMutil.c             void cm_smooth_corner();


RETURNED VALUE

    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES

    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== CM_CORE ROUTINE ===*/

/*******************************************************************/
/*                                                                 */
/*  CORE Model:                                                    */
/*                                                                 */
/*     The core model is designed to operate in one of two modes.  */
/*  The first of these, and the one most likely to be used by      */
/*  the engineer, is a modified version of the pwl model. This     */
/*  behavior occurs when the model is in pwl mode (the default).   */
/*  If the model is set to hyst mode, its behavior mimics that of  */
/*  the hysteresis block. The following provides additional        */
/*  detail:                                                        */
/*                                                                 */
/*                          PWL Mode                               */
/*                                                                 */
/*     In pwl mode, the core model is a modified version of the    */
/*  PWL model...                                                   */
/*  it has a single two-terminal input/output, and accepts as      */
/*  input the mmf value, represented by a voltage. Its output is   */
/*  a flux value, which is represented as a current. Additional    */
/*  inputs include the cross-sectional area of the physical        */
/*  core, and the median length of the core, seen from the         */
/*  perspective of the flux that traverses it.                     */
/*                                                                 */
/*     The core model in pwl mode DOES NOT include hysteresis...   */
/*  current thinking is that such provides                         */
/*  little benefit to the designer, aside from the ability to      */
/*  calculate eddy losses in a modeled device...the nonlinear      */
/*  B vs. H behavior, however, is of great importance.             */
/*                                                                 */
/*     Note that the user must input a piece-wise-linear           */
/*  description, in the form of a series of coordinate B vs. H     */
/*  values, in order to model a particular core material type.     */
/*  Such curves may be found in textbooks, or from manufacturer's  */
/*  databooks. In this model, the "x" values are assumed to        */
/*  represent the magnetic field (H), and the "y" values are       */
/*  assumed to represent the flux density (B).                     */
/*                                                                 */
/*                           Hyst Mode                             */
/*                                                                 */
/*     In hyst mode, the core model is a modified version of the   */
/*  HYST code model...                                             */
/*  it has a single two-terminal input/output, and accepts as      */
/*  input the mmf value, represented by a voltage. Its output is   */
/*  a flux value, which is represented as a current. Additional    */
/*  inputs include the input high and low values for the           */
/*  hysteretic behavior, and the output high and low values.       */
/*  Also, a value of hysteresis must be included, as must an       */
/*  input_domain value, and a fraction value, which tell the model */
/*  whether to interpret the input_domain as an absolute value     */
/*  or as a relative figure.                                       */
/*                                                                 */
/*  When the hyst mode is invoked on the core model, the user is   */
/*  in the position of having to define reasonable values for the  */
/*  upper and lower output limiting values. These can be very      */
/*  difficule to nail down accurately. Current thinking is tha     */
/*  the hysteresis capability will be of only nominal benefit to   */
/*  the engineer, as it will not typically allow for as accurate   */
/*  tailoring of the response as is possible in the pwl mode.      */
/*                                                                 */
/*  4/24/91                                             J.P.Murray */
/*  Last modified: 10/24/91                                        */
/*******************************************************************/

void
cm_core(ARGS)
{

    /*** The following declarations pertain to PWL mode ***/

    int i;                    /* generic loop counter index */
    int size;                 /* size of the x_array        */

    int mode;                 /* mode parameter which determines whether
                                 pwl or hyst will be used in analysis. */


    double input_domain;      /* smoothing range */
    Mif_Value_t *H;           /* pointer to the H-field array */
    Mif_Value_t *B;           /* pointer to the B-field array */
    double lower_seg;         /* x segment below which input resides */
    double upper_seg;         /* x segment above which the input resides */
    double lower_slope;       /* slope of the lower segment */
    double upper_slope;       /* slope of the upper segment */
    double mmf_input;         /* input mmf value */
    double H_input;           /* calculated input H value */
    double B_out;             /* output B value */
    double flux_out;          /* calculated output flux */
    double dout_din;          /* partial derivative of the output wrt input */
    double threshold_lower;   /* value below which the output begins
                                 smoothing */
    double threshold_upper;   /* value above which the output begins smoothing */
    double area;              /* cross-sectional area of the core (in meters)*/
    double length;            /* length of core (in meters) */

    Mif_Complex_t ac_gain;

    char *limit_error="\n***ERROR***\nCORE: Violation of 50% rule in breakpoints!\n";


    /*** The following declarations pertain to HYSTERESIS mode... ***/

    double
        in,                       /* input to hysteresis block */
        out,                      /* output from hysteresis block */
        in_low,                   /* lower input value for hyst=0 at which
                                     the transfer curve changes from constant
                                     to linear */
        in_high,                  /* upper input value for hyst=0 at which
                                     the transfer curve changes from constant
                                     to linear */
        hyst,                     /* the hysteresis value (see above diagram) */
        out_lower_limit,          /* the minimum output value from the block */
        out_upper_limit,          /* the maximum output value from the block */
        slope,                    /* calculated rise and fall slope for the block */
        pout_pin,                 /* partial derivative of output w.r.t. input */
        x_rise_linear,            /* = in_low  + hyst */
        x_rise_zero,              /* = in_high + hyst */
        x_fall_linear,            /* = in_high - hyst */
        x_fall_zero;              /* = in_low  - hyst */

    Boolean_t
        *hyst_state,              /* TRUE => input is on lower leg of
                                     hysteresis curve, between -infinity
                                     and in_high + hyst.
                                     FALSE => input is on upper leg
                                     of hysteresis curve, between
                                     in_low - hyst and +infinity */
        *old_hyst_state;          /* previous value of *hyst_state */

    /* Retrieve mode parameter... */

    mode = PARAM(mode);


    /** Based on mode value, switch to the appropriate model code... **/

    if (HYSTERESIS != mode) {   /******** pwl mode *****************/

        /* Retrieve frequently used parameters... */

        input_domain = PARAM(input_domain);
        area = PARAM(area);
        length = PARAM(length);

        size = PARAM_SIZE(H_array);

        H = (Mif_Value_t*) &PARAM(H_array[0]);
        B = (Mif_Value_t*) &PARAM(B_array[0]);

        /* See if input_domain is absolute...if so, test against   */
        /* breakpoint segments for violation of 50% rule...        */
        if (PARAM(fraction) == MIF_FALSE)
            for (i = 0; i < size - 1; i++)
                if ((H[i+1].rvalue - H[i].rvalue) < 2.0 * input_domain) {
                    cm_message_send(limit_error);
                    return;
                }

        /* Retrieve mmf_input value. */
        mmf_input = INPUT(mc);

        /* Calculate H_input value from mmf_input... */
        H_input = mmf_input / length;

        /* Determine segment boundaries within which H_input resides */

        if (H_input <= (H[1].rvalue + H[0].rvalue) / 2.0) {/*** H_input below lowest midpoint ***/

            dout_din = (B[1].rvalue - B[0].rvalue) / (H[1].rvalue - H[0].rvalue);
            B_out = B[0].rvalue + (H_input - H[0].rvalue) * dout_din;

        } else if (H_input >= (H[size-2].rvalue + H[size-1].rvalue) / 2.0) {

            /*** H_input above highest midpoint ***/
            dout_din = (B[size-1].rvalue - B[size-2].rvalue) / (H[size-1].rvalue - H[size-2].rvalue);
            B_out = B[size-1].rvalue + (H_input - H[size-1].rvalue) * dout_din;

        } else { /*** H_input within bounds of end midpoints... ***/

            /*** must determine position progressively & then ***/
            /*** calculate required output. ***/

            dout_din = NAN;
            B_out    = NAN;

            for (i = 1; i < size; i++)
                if (H_input < (H[i].rvalue + H[i+1].rvalue) / 2.0) {
                    /* approximate position known... */

                    lower_seg = (H[i].rvalue - H[i-1].rvalue);
                    upper_seg = (H[i+1].rvalue - H[i].rvalue);

                    /* Calculate input_domain about this region's breakpoint.*/

                    if (PARAM(fraction) == MIF_TRUE) {  /* Translate input_domain */
                        /* into an absolute....   */
                        if (lower_seg <= upper_seg)            /* Use lower  */
                            /* segment    */
                            /* for % calc.*/
                            input_domain = input_domain * lower_seg;
                        else                                   /* Use upper  */
                            /* segment    */
                            /* for % calc.*/
                            input_domain = input_domain * upper_seg;
                    }

                    /* Set up threshold values about breakpoint... */
                    threshold_lower = H[i].rvalue - input_domain;
                    threshold_upper = H[i].rvalue + input_domain;

                    /* Determine where H_input is within region & determine     */
                    /* output and partial values....                            */
                    if (H_input < threshold_lower) { /* Lower linear region     */

                        dout_din = (B[i].rvalue - B[i-1].rvalue) / lower_seg;
                        B_out = B[i].rvalue + (H_input - H[i].rvalue) * dout_din;

                    } else if (H_input < threshold_upper) { /* Parabolic region */

                        lower_slope = (B[i].rvalue - B[i-1].rvalue) / lower_seg;
                        upper_slope = (B[i+1].rvalue - B[i].rvalue) / upper_seg;
                        cm_smooth_corner(H_input, H[i].rvalue, B[i].rvalue, input_domain,
                                         lower_slope, upper_slope, &B_out, &dout_din);

                    } else {      /* Upper linear region */

                        dout_din = (B[i+1].rvalue - B[i].rvalue) / upper_seg;
                        B_out = B[i].rvalue + (H_input - H[i].rvalue) * dout_din;

                    }

                    break;  /* Break search loop...H_input has been found, */
                    /* and B_out and dout_din have been assigned. */
                }
        }

        /* Calculate value of flux_out... */
        flux_out = B_out * area;

        /* Adjust dout_din value to reflect area and length multipliers... */
        dout_din = dout_din * area / length;

        if (ANALYSIS != MIF_AC) {        /* Output DC & Transient Values */
            OUTPUT(mc) = flux_out;
            PARTIAL(mc, mc) = dout_din;
        } else {                    /* Output AC Gain */
            ac_gain.real = dout_din;
            ac_gain.imag = 0.0;
            AC_GAIN(mc, mc) = ac_gain;
        }

    } else {                    /******** hysteresis mode ******************/

        /** Retrieve frequently used parameters... **/

        in_low = PARAM(in_low);
        in_high = PARAM(in_high);
        hyst = PARAM(hyst);
        out_lower_limit = PARAM(out_lower_limit);
        out_upper_limit = PARAM(out_upper_limit);
        input_domain = PARAM(input_domain);

        /** Calculate Hysteresis Linear Region Slopes & Derived Values **/

        /* Define slope of rise and fall lines when not being smoothed */

        slope = (out_upper_limit - out_lower_limit) / (in_high - in_low);

        x_rise_linear = in_low + hyst;    /* Breakpoint - x rising to
                                             linear region */
        x_rise_zero   = in_high + hyst;   /* Breakpoint - x rising to
                                             zero-slope (out_upper_limit) */
        x_fall_linear = in_high - hyst;   /* Breakpoint - x falling to
                                             linear region */
        x_fall_zero   = in_low - hyst;    /* Breakpoint - x falling to
                                             zero-slope (out_lower_limit) */

        /* Set range to absolute value */
        if (PARAM(fraction) == MIF_TRUE)
            input_domain = input_domain * (in_high - in_low);

        /** Retrieve frequently used inputs... **/

        in = INPUT(mc);

        /** Test for INIT; if so, allocate storage, otherwise, retrieve
            previous timepoint value for output...     **/

        /* First pass...allocate storage for previous state.   */
        /* Also, calculate roughly where the current output    */
        /* will be and use this value to define current state. */
        if (INIT == 1) {

            cm_analog_alloc(TRUE, sizeof(Boolean_t));

            hyst_state     = (Boolean_t *) cm_analog_get_ptr(TRUE, 0);
            old_hyst_state = (Boolean_t *) cm_analog_get_ptr(TRUE, 1);

            if (in < x_rise_zero + input_domain)  /* Set state to X_RISING */
                *old_hyst_state = X_RISING;
            else
                *old_hyst_state = X_FALLING;

        } else { /* Allocation not necessary...retrieve previous values */

            hyst_state = (Boolean_t *) cm_analog_get_ptr(TRUE, 0);  /* Set out pointer to current
                                                                       time storage */
            old_hyst_state = (Boolean_t *) cm_analog_get_ptr(TRUE, 1);  /* Set old-output-state pointer
                                                                           to previous time storage */
        }

        /** Set *hyst_out = *old_hyst_out, unless changed below...
            we don't need the last iteration value of *hyst_state.  **/

        *hyst_state = *old_hyst_state;

        /*** Calculate value of hyst_state, pout_pin.... ***/

        if (*old_hyst_state == X_RISING) { /* Assume calculations on lower  */
            /* hysteresis section (x rising) */

            if (in <= x_rise_linear - input_domain) { /* Output @ lower limit */

                out = out_lower_limit;
                pout_pin = 0.0;

            } else if (in <= x_rise_linear + input_domain) { /* lower smoothing region */

                cm_smooth_corner(in, x_rise_linear, out_lower_limit, input_domain,
                                 0.0, slope, &out, &pout_pin);

            } else if (in <= x_rise_zero - input_domain) { /* Rising linear region */

                out = (in - x_rise_linear)*slope + out_lower_limit;
                pout_pin = slope;

            } else if (in <= x_rise_zero + input_domain) { /* Upper smoothing region */

                cm_smooth_corner(in, x_rise_zero, out_upper_limit, input_domain,
                                 slope, 0.0, &out, &pout_pin);

            } else { /* input has transitioned to X_FALLING region... */

                out = out_upper_limit;
                pout_pin = 0.0;
                *hyst_state = X_FALLING;

            }

        } else {  /* Assume calculations on upper hysteresis section (x falling) */

            if ( in >= x_fall_linear + input_domain ) { /* Output @ upper limit */

                out = out_upper_limit;
                pout_pin = 0.0;

            } else if ( in >= x_fall_linear - input_domain ) { /* Upper smoothing region */

                cm_smooth_corner(in, x_fall_linear, out_upper_limit, input_domain,
                                 slope, 0.0, &out, &pout_pin);

            } else if (in >= x_fall_zero + input_domain) { /* Falling linear region */

                out = (in - x_fall_zero)*slope + out_lower_limit;
                pout_pin = slope;

            } else if (in >= x_fall_zero - input_domain) { /* Lower smoothing region */

                cm_smooth_corner(in, x_fall_zero, out_lower_limit, input_domain,
                                 0.0, slope, &out, &pout_pin);

            } else { /* input has transitioned to X_RISING region... */

                out = out_lower_limit;
                pout_pin = 0.0;
                *hyst_state = X_RISING;

            }
        }

        if (ANALYSIS != MIF_AC) {     /* DC & Transient Analyses */
            OUTPUT(mc) = out;
            PARTIAL(mc, mc) = pout_pin;
        } else {                      /* AC Analysis */
            ac_gain.real = pout_pin;
            ac_gain.imag = 0.0;
            AC_GAIN(mc, mc) = ac_gain;
        }
    }
}

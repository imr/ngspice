/* ===========================================================================
FILE    cfunc.mod

MEMBER OF process XSPICE

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503

AUTHORS

    9/12/91  Bill Kuhn

MODIFICATIONS

    <date> <person name> <nature of modifications>

SUMMARY

    This file contains the definition of an inductor code model
    with current initial conditions.

INTERFACES

    cm_inductor()

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

=========================================================================== */


#define LI  0


void cm_inductor (ARGS)
{
    Complex_t   ac_gain;
    double      partial;
    double      ramp_factor;
    double      *li;

    /* Get the ramp factor from the .option ramptime */
    ramp_factor = cm_analog_ramp_factor();

    /* Initialize/access instance specific storage for capacitor voltage */
    if(INIT) {
        cm_analog_alloc(LI, sizeof(double));
        li = (double *) cm_analog_get_ptr(LI, 0);
        *li = PARAM(ic) * ramp_factor;
    }
    else {
        li = (double *) cm_analog_get_ptr(LI, 0);
    }

    /* Compute the output */
    if(ANALYSIS == DC) {
        OUTPUT(ind) = PARAM(ic) * ramp_factor;
        PARTIAL(ind, ind) = 0.0;
    }
    else if(ANALYSIS == AC) {
        ac_gain.real = 0.0;
        ac_gain.imag = 1.0 * RAD_FREQ * PARAM(l);
        AC_GAIN(ind, ind) = ac_gain;
    }
    else if(ANALYSIS == TRANSIENT) {
        if(ramp_factor < 1.0) {
            *li = PARAM(ic) * ramp_factor;
            OUTPUT(ind) = *li;
            PARTIAL(ind, ind) = 0.0;
        }
        else {
            cm_analog_integrate(INPUT(ind) / PARAM(l), li, &partial);
            partial /= PARAM(l);
            OUTPUT(ind) = *li;
            PARTIAL(ind, ind) = partial;
        }
    }
}


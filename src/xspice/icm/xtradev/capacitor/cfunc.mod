/* ===========================================================================
FILE    capacitor/cfunc.mod

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

    This file contains the definition of a capacitor code model
    with voltage type initial conditions.

INTERFACES

    cm_capacitor()

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

=========================================================================== */


#define VC  0


void cm_capacitor (ARGS)
{
    Complex_t   ac_gain;
    double      partial;
    double      ramp_factor;
    double      *vc;


    /* Get the ramp factor from the .option ramptime */
    ramp_factor = cm_analog_ramp_factor();

    /* Initialize/access instance specific storage for capacitor voltage */
    if(INIT) {
        cm_analog_alloc(VC, sizeof(double));
        vc = (double *) cm_analog_get_ptr(VC, 0);
        *vc = PARAM(ic) * cm_analog_ramp_factor();
    }
    else {
        vc = (double *) cm_analog_get_ptr(VC, 0);
    }

    /* Compute the output */
    if(ANALYSIS == DC) {
        OUTPUT(cap) = PARAM(ic) * ramp_factor;
        PARTIAL(cap, cap) = 0.0;
    }
    else if(ANALYSIS == AC) {
        ac_gain.real = 0.0;
        ac_gain.imag = -1.0 / RAD_FREQ / PARAM(c);
        AC_GAIN(cap, cap) = ac_gain;
    }
    else if(ANALYSIS == TRANSIENT) {
        if(ramp_factor < 1.0) {
            *vc = PARAM(ic) * ramp_factor;
            OUTPUT(cap) = *vc;
            PARTIAL(cap, cap) = 0.0;
        }
        else {
            cm_analog_integrate(INPUT(cap) / PARAM(c), vc, &partial);
            partial /= PARAM(c);
            OUTPUT(cap) = *vc;
            PARTIAL(cap, cap) = partial;
        }
    }
}


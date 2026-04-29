/*
================================================================================

FILE ota/cfunc.mod

Public Domain

AUTHORS

    20 Mar 2026     Seth Hillbrand

SUMMARY

    This file contains the model-specific routines for the OTA
    (Operational Transconductance Amplifier) code model.

    DC/TRAN: Vout as current = gm * (Vp - Vn)
    AC:      Complex gain = gm
    NOISE:   Programmatic noise with en, in_noise, enk, ink, incm, incmk.

    Parameter naming follows LTSPICE convention for OTA compatibility.

================================================================================
*/

#include <stdlib.h>


void cm_ota(ARGS)
{
    double gm_val = PARAM(gm);
    double rout = PARAM(rout);
    double rin = PARAM(rin);

    Mif_Complex_t ac_gain;

    if (ANALYSIS == NOISE) {
        /* Register noise sources on every call (N_OPEN and N_CALC).
         * During N_OPEN: registers and returns index.
         * During N_CALC: returns same sequential index. */
        int src_en_w = cm_noise_add_source("en_white", 1, 0, MIF_NOISE_CURRENT);
        int src_en_f = cm_noise_add_source("en_flicker", 1, 0, MIF_NOISE_CURRENT);
        int src_in_w = cm_noise_add_source("in_white", 0, 0, MIF_NOISE_CURRENT);
        int src_in_f = cm_noise_add_source("in_flicker", 0, 0, MIF_NOISE_CURRENT);

        /* Common-mode current noise: independent sources from each input pin to ground */
        int src_icm_pw = cm_noise_add_source("incm_p_white", 0, 0, MIF_NOISE_CURRENT_POS);
        int src_icm_pf = cm_noise_add_source("incm_p_flicker", 0, 0, MIF_NOISE_CURRENT_POS);
        int src_icm_nw = cm_noise_add_source("incm_n_white", 0, 0, MIF_NOISE_CURRENT_NEG);
        int src_icm_nf = cm_noise_add_source("incm_n_flicker", 0, 0, MIF_NOISE_CURRENT_NEG);

        if (!mif_private->noise->registering) {
            double en = PARAM(en);
            double in_n = PARAM(in_noise);
            double enk_val = PARAM(enk);
            double ink_val = PARAM(ink);
            double incm = PARAM(incm);
            double incmk_val = PARAM(incmk);
            double f = NOISE_FREQ;

            /* en referred to output as current noise: (en * gm)^2 A^2/Hz */
            NOISE_DENSITY(src_en_w) = en * en * gm_val * gm_val;
            NOISE_DENSITY(src_en_f) = (enk_val > 0 && f > 0) ?
                en * en * gm_val * gm_val * enk_val / f : 0.0;

            /* in as differential current noise at input: in^2 A^2/Hz */
            NOISE_DENSITY(src_in_w) = in_n * in_n;
            NOISE_DENSITY(src_in_f) = (ink_val > 0 && f > 0) ?
                in_n * in_n * ink_val / f : 0.0;

            /* incm as current noise from each input pin to ground */
            NOISE_DENSITY(src_icm_pw) = incm * incm;
            NOISE_DENSITY(src_icm_pf) = (incmk_val > 0 && f > 0) ?
                incm * incm * incmk_val / f : 0.0;
            NOISE_DENSITY(src_icm_nw) = incm * incm;
            NOISE_DENSITY(src_icm_nf) = (incmk_val > 0 && f > 0) ?
                incm * incm * incmk_val / f : 0.0;
        }

        return;
    }

    if (ANALYSIS != MIF_AC) {
        double v_in = INPUT(inp);
        double v_out = INPUT(out);

        OUTPUT(out) = gm_val * v_in + v_out / rout;
        PARTIAL(out, inp) = gm_val;
        PARTIAL(out, out) = 1.0 / rout;

        OUTPUT(inp) = v_in / rin;
        PARTIAL(inp, inp) = 1.0 / rin;
    }
    else {
        Mif_Complex_t ac_rout;
        Mif_Complex_t ac_rin;

        ac_gain.real = gm_val;
        ac_gain.imag = 0.0;
        AC_GAIN(out, inp) = ac_gain;

        ac_rout.real = 1.0 / rout;
        ac_rout.imag = 0.0;
        AC_GAIN(out, out) = ac_rout;

        ac_rin.real = 1.0 / rin;
        ac_rin.imag = 0.0;
        AC_GAIN(inp, inp) = ac_rin;
    }
}

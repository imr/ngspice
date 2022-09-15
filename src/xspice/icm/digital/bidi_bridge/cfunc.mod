/*  -*- mode: C;-*-  (emacs magic)
 *
 * Simple bidirectional digital/analog bridge.
 * Giles Atkinson 2022
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <string.h>

/* These define driver cut-off: use PARAMs? */

#define OFF_LOW  0.7
#define OFF_HIGH 0.3

/* Structure for analogue state: input and output. */

struct a_data {
    Digital_t i;         // Previous output drive
    double    svoc;      // Previous scaled open-circuit estimate
    double    o;         // Previous output
};

/* Structure for digital state: input and output. */

struct d_data {
    Digital_t i, n, o;   // Input, new output, previous output
    double    i_changed; // Time of first input change.
};

/* Called at end to free memory. */

void cm_bidi_bridge(ARGS)
{
    Digital_State_t  atod;
    struct d_data   *d;
    struct a_data   *a, *olda;
    int              i, size, dsize, strength, dir_param;
    double           delta, rise_delay, fall_delay, t_rise, t_fall;


    size = PORT_SIZE(a);
    dsize = PORT_NULL(dir) ? 0 : PORT_SIZE(dir);

    if (INIT) {
        /* Allocate storage for previous values on both sides */

        cm_analog_alloc(0, size * (int) sizeof (struct a_data));
        a = (struct a_data *)cm_analog_get_ptr(0, 1);
        cm_event_alloc(0, size * (int) sizeof (struct d_data));
        d = (struct d_data *)cm_event_get_ptr(0, 0);

        for (i = 0; i < size; ++i) {
            LOAD(d[i]) = PARAM(input_load);
            a[i].o = 0.0;
            a[i].i.state = UNKNOWN;
            a[i].i.strength = HI_IMPEDANCE;

            /* Force port initialisation on first event call. */

            d[i].i.state = UNKNOWN;
            d[i].i.strength = UNDETERMINED;
            d[i].o.state = (Digital_State_t)(UNKNOWN + 1); // Force update
            d[i].n.state = UNKNOWN;
            d[i].i_changed = -1.0;
            OUTPUT_STATE(d[i]) = UNKNOWN;
            OUTPUT_STRENGTH(d[i]) = d[i].o.strength = HI_IMPEDANCE;
        }
        for (i = 0; i < dsize; ++i)
            LOAD(dir[i]) = PARAM(input_load);
    }

    if (dsize > size)
        dsize = size;
    strength = PARAM(strength);
    dir_param = PARAM(direction);
    t_rise = PARAM(t_rise);
    t_fall = PARAM(t_fall);
    d = (struct d_data *)cm_event_get_ptr(0,0);

    if (CALL_TYPE == ANALOG) {
        double in_high, in_low;
        double out_high, out_low, drive_high, drive_low, r_high, r_low;
        double r_stl, r_sth;
        int    smooth; // Smoothing level 0-2 (default 0).

        a = (struct a_data *)cm_analog_get_ptr(0,0);
        olda = (struct a_data *)cm_analog_get_ptr(0, 1);
        in_high = PARAM(in_high);
        in_low = PARAM(in_low);
        out_high = PARAM(out_high);
        out_low = PARAM(out_low);
        drive_low = PARAM(drive_low);
        r_stl = PARAM(r_stl);
        drive_high = PARAM(drive_high);
        r_sth = PARAM(r_sth);
        r_low = PARAM(r_low);
        r_high = PARAM(r_high);
        smooth = PARAM(smooth);

        for (i = 0; i < size; ++i) {
            double svoc, voc; // Notional open-circuit voltage scaled/actual.
            double in, out;

            /* Determine direction. */

            in = INPUT(a[i]);
            atod = (Digital_State_t)dir_param;
            if (atod == UNKNOWN)
                atod =  (i < dsize) ? INPUT_STATE(dir[i]) : UNKNOWN;

            /* Analog to digital. */

            if (atod != ZERO) {
                Digital_State_t new;

                /* Determine digital output state. */

                if (in_high < in_low) {   // Hysteresis
                    if (in > in_low)
                        new = ONE;
                    else if (in < in_high)
                        new = ZERO;
                    else
                        new = d[i].o.state; // Same as before
                } else {
                    if (in < in_low)
                        new = ZERO;
                    else if (in > in_high)
                        new = ONE;
                    else
                        new = UNKNOWN;
                }
                if (new != d[i].n.state) {
                    /* Output change, schedule EVENT call. */

                    d[i].n.state = new;
                    cm_event_queue(TIME);
                }
            }

            /* Digital to analog. */

            if (atod == ONE) {
                out = 0.0;   /* AtoD, no analogue output. */
                svoc = 0.5;
            } else {
                double    target, iota, interval[2], range, partial;
                int       step, step_count;
                Digital_t drive, *dp;

                /* Find analogue output current. */

                if (atod == UNKNOWN) {
                    /* What is the external drive on the digital node? */

                    drive.state = UNKNOWN; drive.strength = HI_IMPEDANCE;
                    if (!cm_probe_node(1, (unsigned int)i, &drive))
                        cm_message_printf("Can not probe port d[%d].", i);
                } else {
                    drive = *(Digital_t *)INPUT(d[i]);
                }
                a[i].i = drive;

                /* Has the input changed during this timestep? */

                iota = (T(0) - T(1)) * 1e-6; // Ignorable
                if (T(0) - d[i].i_changed < iota) {
                    /* Previous input value in force for whole step. */

                    step_count = 1;
                    step = 0;
                    interval[0] = T(0) - T(1);
                } else if (d[i].i_changed - T(1) < iota) {
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
                    interval[0] = d[i].i_changed - T(1);
                    interval[1] = T(0) - d[i].i_changed;
                }

                out = olda[i].o;
                svoc = olda[i].svoc;
                for (; step < step_count; ++step) {
                    double change, tv;
                    double max_high, max_low;

                    if (step == 0) {
                        dp = &olda[i].i;
                    } else {
                        dp = &drive;
                    }

                    /* Calculate new value for open-circuit output voltage. */

                    change = T(0) - T(1);
                    switch (dp->state) {
                    case ZERO:
                        svoc -= change / t_fall;
                        if (svoc < 0.0)
                            svoc = 0.0;
                        break;
                    case ONE:
                        svoc += change / t_rise;
                        if (svoc > 1.0)
                            svoc = 1.0;
                        break;
                    default:
                        if (dp->strength == HI_IMPEDANCE) {
                            svoc = 0.5; // Keeps compiler happy
                        } else {
                            /* Assume both drivers are on.  */

                            if (dp->strength == STRONG) {
                                if (drive_high > drive_low)
                                    tv = 1.0;
                                else if (drive_high > drive_low)
                                    tv = 0.0;
                                else
                                    tv = 0.5;
                            } else {
                                tv = r_low / (r_low + r_high);
                            }
                            if (svoc < tv) {
                                svoc += change / t_rise;
                                if (svoc > tv)
                                    svoc = tv;
                            } else {
                                svoc -= change / t_fall;
                                if (svoc < tv)
                                    svoc = tv;
                            }
                        }
                    }
                    if (smooth > 0) {
                        cm_smooth_discontinuity(svoc, 0.0, 0.0, 1.0, 1.0,
                                                &voc, &tv); // tv is dummy.
                    } else {
                        voc = svoc;
                    }

                    /* Available current depends on svoc (driver cut-off). */

                    max_high = drive_high;
                    max_high *= (voc - OFF_HIGH) / (1 - OFF_HIGH);
                    if (max_high < 0.0)
                        max_high = 0.0;
                    if (smooth > 1) {
                        cm_smooth_discontinuity(max_high, 0.0, 0.0,
                                                drive_high, drive_high,
                                                &max_high, &tv);
                    }
                    max_low = drive_low * (OFF_LOW - voc) / OFF_LOW;
                    if (max_low < 0.0)
                        max_low = 0.0;
                    if (smooth > 1) {
                        cm_smooth_discontinuity(max_low, 0.0, 0.0,
                                                drive_low, drive_low,
                                                &max_low, &tv);
                    }

                    /* Convert to voltage. */

                    voc = out_low + (out_high - out_low) * voc;
                    target = 0.0;
                    partial = 0.0;

                    /* Calculate new value for output current. */

                    switch (dp->strength) {
                    case STRONG:
                        range = drive_high + drive_low;
                        switch (dp->state) {
                        case ZERO:
                            target = (in - voc) / r_stl;
                            partial = 1.0 / r_stl;
                            break;
                        case ONE:
                            target = (in - voc) / r_sth;
                            partial = 1.0 / r_sth;
                            break;
                        case UNKNOWN:
                            /* Assume both drivers are on.  */

                            target = (in - voc) *
                                ((r_stl + r_sth) / (r_stl * r_sth));
                            partial = (r_stl + r_sth) / (r_stl * r_sth);
                            break;
                        }
                        if (target > max_low) {
                            target = max_low;
                            partial = 0.0;
                        } else if (target < -max_high) {
                            target = -max_high;
                            partial = 0.0;
                        }
                        if (smooth > 2) {
                            cm_smooth_discontinuity(target,
                                                    -max_high, -max_high,
                                                    max_low, max_low,
                                                    &target, &tv);
                        }
                        break;
                    case RESISTIVE:
                    case UNDETERMINED:   // Who knows?
                        range = out_high / r_high + out_low / r_low;
                        switch (dp->state) {
                        case ZERO:
                            if (in < voc) {
                                target = 0.0;
                            } else {
                                target = (in - voc) / r_low;
                                partial = 1.0 / r_low;
                            }
                            break;
                        case ONE:
                            if (in > voc) {
                                target = 0.0;
                            } else {
                                target = (in - voc) / r_high;
                                partial = 1.0 / r_high;
                            }
                            break;
                        case UNKNOWN:
                            if (in < out_low) {
                                target = (in - out_high) / r_high;
                                partial = 1.0 / r_high;
                            } else if (in >= out_high) {
                                target = (in - out_low) / r_low;
                                partial = 1.0 / r_low;
                            } else {
                                /* Both drivers on. */

                                partial = 1.0 / r_low + 1.0 / r_high;
                                target = (in - voc ) * partial;
                            }
                            break;
                        }
                        break;
                    case HI_IMPEDANCE:
                    default:
                        range = fmax(drive_high, drive_low);
                        target = 0.0;
                        break;
                    }

                    /* Can the transition complete in available time? */

                    delta = target - out; // Current
                    delta *=
                        ((delta > 0) ? t_rise : t_fall) / range; // Time
                    if (delta < 0)
                        delta = -delta;
                    if (delta > interval[step]) {
                        out += (target - out) * interval[step] / delta;
                    } else {
                        /* Transition complete. */

                        out = target;
                    }
                }
                if (partial != 0.0)
                    PARTIAL(a[i], a[i]) = partial;
            }
            OUTPUT(a[i]) = a[i].o = out;
            a[i].svoc = svoc;
        }
    } else {
        /* Digital. */

        rise_delay = PARAM(rise_delay);
        fall_delay = PARAM(fall_delay);

        for (i = 0; i < size; ++i) {
            /* Determine direction. */

            atod = (Digital_State_t)dir_param;
            if (atod == UNKNOWN)
                atod =  (i < dsize) ? INPUT_STATE(dir[i]) : UNKNOWN;

            if (atod != ONE && (INPUT_STATE(d[i]) != d[i].i.state ||
                                INPUT_STRENGTH(d[i]) != d[i].i.strength)) {
                double transition;

                /* Digital input changed, request break. */

                d[i].i = *(Digital_t *)INPUT(d[i]);
                cm_analog_set_temp_bkpt(TIME);
                d[i].i_changed = TIME;

                /* Do not let big timesteps smear the transition (it could be
                 * much faster with current output into high impedance.
                 */

                transition = (d[i].i.state == ZERO) ? t_fall :
                    (d[i].i.state == ONE) ? t_rise : fmin(t_rise, t_fall);
                cm_analog_set_perm_bkpt(TIME + transition * 1.0001);
            }

            /* Check for output change from analogue. */

            if (atod == ZERO) {
                if (d[i].o.strength != HI_IMPEDANCE) {
                    OUTPUT_STATE(d[i]) = d[i].o.state = UNKNOWN;
                    OUTPUT_STRENGTH(d[i]) = d[i].o.strength = HI_IMPEDANCE;
                    OUTPUT_DELAY(d[i]) = fmax(rise_delay, fall_delay);
                } else {
                    OUTPUT_CHANGED(d[i]) = FALSE;
                }
            } else {
                if (d[i].o.state != d[i].n.state) {
                    double delay;

                    OUTPUT_STATE(d[i]) = d[i].o.state = d[i].n.state;
                    OUTPUT_STRENGTH(d[i]) = d[i].o.strength = strength;
                    switch (d[i].o.state) {
                    case ZERO:
                        delay = fall_delay;
                        break;
                    case ONE:
                        delay = rise_delay;
                        break;
                    default:
                    case UNKNOWN:
                        delay = fmin(rise_delay, fall_delay);
                        break;
                    }
                    OUTPUT_DELAY(d[i]) = delay;
                } else {
                    OUTPUT_CHANGED(d[i]) = FALSE;
                }
            }
        }
    }
}

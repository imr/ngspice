/* XSPICE code model for the Controlled Digital Oscillator.
 * This is a complete redesign of the original version by the
 * Georgia Tech team, as a fix for ngspice Bug #629 - "XSPICE d_osc failures".
 */

#include <stdlib.h>

#define FACTOR 0.75     // Controls timing of next scheduled call. */

/* PWL table entry. */

struct pwl {
    double ctl, freq;
};

/* Called at end to free memory. */

static void cm_d_osc_callback(ARGS, Mif_Callback_Reason_t reason)
{
    struct panel_instance *instance;

    if (reason == MIF_CB_DESTROY) {
        struct pwl *table = STATIC_VAR(locdata);

        if (table)
            free(table);
        STATIC_VAR(locdata) = NULL;
    }
}

/* Get the current period. */

static double get_period(double ctl, struct pwl *table, int csize)
{
    double f;
    int    i;

    for (i = 0; i < csize; ++i) {
         if (table[i].ctl > ctl)
             break;
    }

    /* Interpolation outside input range continues slope. */

    if (i > 0) {
       if (i == csize)
           i -= 2;
       else
           i--;
    }
    f = table[i].freq +
            (ctl - table[i].ctl) * ((table[i + 1].freq - table[i].freq) /
                                    (table[i + 1].ctl - table[i].ctl));
     return 1.0 / f;
}

/* The state data. */

struct state {
    double          last_time; // Time of last output change.
    Digital_State_t last;      // Last value output.
};

/* The code-model function. */

void cm_d_osc(ARGS)
{
    struct pwl   *table;
    struct state *state;
    double        ctl, delta, when;
    int           csize, i;

    CALLBACK = cm_d_osc_callback;

    csize = PARAM_SIZE(cntl_array);
    if (INIT) {

        /* Validate PWL table. */

        for (i = 0; i < csize - 1; ++i) {
            if (PARAM(cntl_array[i]) >= PARAM(cntl_array[i + 1]))
                break;
        }

        if (i < csize - 1 || csize != PARAM_SIZE(freq_array)) {
            cm_message_send("Badly-formed control table");
            STATIC_VAR(locdata) = NULL;
            return;
        }

        /* Allocate PWL table. */

        table = malloc(csize * sizeof (struct pwl));
        STATIC_VAR(locdata) = table;
        if (!table)
            return;

        for (i = 0; i < csize; ++i) {
            table[i].ctl = PARAM(cntl_array[i]);
            table[i].freq = PARAM(freq_array[i]);
            if (table[i].freq <= 0) {
                cm_message_printf("Error: frequency %g is not positve, "
                                  "value replaced by 1e-16.",
                                  table[i].freq);
                table[i].freq = 1.0e-16;
            }
        }

        /* Allocate state data. */

        cm_event_alloc(0, sizeof (struct state));
        return;
    }

    table = STATIC_VAR(locdata);
    if (!table)
         return;
    state = (struct state *)cm_event_get_ptr(0, 0);

    if (CALL_TYPE != EVENT) {
        if (TIME == 0.0) {
            double phase;

            /* Set initial output and state data. */

            ctl = INPUT(cntl_in);
            delta = get_period(ctl, table, csize);

            phase = PARAM(init_phase);
            phase /= 360.0;
            if (phase < 0.0)
                phase += 1.0;

            /* When would a hypothetical previous transition have been? */

            state->last_time = delta * (1.0 - PARAM(duty_cycle) - phase);
            if (state->last_time < 0.0) {
                state->last = ONE;
            } else {
                state->last = ZERO;
                state->last_time = -delta * phase;
            }
        }
        return;
    }

    /* Event call; either one requested previously or just before
     * a time-step is accepted.
     */

     if (TIME == 0.0) {
         OUTPUT_STATE(out) = state->last;
         OUTPUT_STRENGTH(out) = STRONG;
         return;
     }

     /* When is the next transition due? */

     ctl = INPUT(cntl_in);
     delta = get_period(ctl, table, csize);
     if (state->last)
         delta *= PARAM(duty_cycle);
     else
         delta *= (1.0 - PARAM(duty_cycle));
     when = state->last_time + delta;

     if (TIME >= when) {
         /* If the frequency rose rapidly, the transition has been missed.
          * Force a shorter time-step and schedule then.
          */

         cm_analog_set_temp_bkpt(state->last_time + FACTOR * delta);
         OUTPUT_CHANGED(out) = FALSE;
         return;
     }

     if (TIME >= state->last_time + FACTOR * delta) {
         /* TIME is reasonably close to transition time.  Request output. */

         state->last_time = when;
         state->last ^= ONE;
         OUTPUT_STATE(out) = state->last;
         OUTPUT_STRENGTH(out) = STRONG;
         OUTPUT_DELAY(out) = when - TIME;

         /* Request a call in the next half-cycle. */

         cm_event_queue(when + FACTOR * delta);
     } else {
         OUTPUT_CHANGED(out) = FALSE;

         if (TIME < state->last_time) {
             /* Output transition pending, nothing to do. */

             return;
         } else {
             /* Request a call nearer to transition time. */

             cm_event_queue(state->last_time + FACTOR * delta);
         }
     }
}

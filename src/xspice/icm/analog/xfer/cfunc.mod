/* Transfer function block for AC simulation, based on s_xfer code model. */

#include <stdlib.h>

#define PI 3.141592653589793238462643383279502884197

/* How the table information is stored internally. */

struct data_pt {
    double        f; /* Frequency, radians/sec. */
    Mif_Complex_t s; /* The S-parameter. */
};

static void cleanup(ARGS, Mif_Callback_Reason_t reason)
{
    struct data_pt *table;

    switch (reason) {
    case MIF_CB_DESTROY:
        table = (struct data_pt *)STATIC_VAR(table);
        if (table) {
            free(table);
            STATIC_VAR(table) = NULL;
        }
        break;
    }
}

void cm_xfer(ARGS)  /* structure holding parms, inputs, outputs, etc.     */
{
    struct data_pt *table;
    Mif_Complex_t   ac_gain;
    double          factor;
    int             span, size, i;

    span = PARAM(span);
    if (INIT) {
        Mif_Boolean_t ri, db, rad;
        int           offset, bad = 0, j;

        /* Validate table. */

        offset = PARAM(offset);
        size = PARAM_SIZE(table);
        bad = size % span;
        if (!bad) {
            for (i = 0; i < size - span; i += span) {
                if (PARAM(table[i]) < 0 ||
                    PARAM(table[i + span]) < PARAM(table[i])) {
                    bad = 1;
                    break;
                }
            }
        }
        if (bad) {
            cm_message_send("Warning: badly formed table.");
            return;
        }

        /* Allocate the internal table. */

        size /= span;
        table = (struct data_pt *)calloc(size, sizeof(struct data_pt));
        STATIC_VAR(table) = table;
        CALLBACK = cleanup;

        /* Fill it. */

	ri = PARAM(r_i);
	db = PARAM(db);
	rad = PARAM(rad);
        for (i = 0, j = 0; i < size; i++, j += span) {
            table[i].f = PARAM(table[j]) * 2.0 * PI;
	    if (ri) {
                table[i].s.real = PARAM(table[j + offset]);
                table[i].s.imag = PARAM(table[j + offset + 1]);
            } else {
                double phase, mag;

                mag = PARAM(table[j + offset]);
                if (db)
                    mag = pow(10, mag / 20);
                phase = PARAM(table[j + offset + 1]);
                if (!rad)
                    phase *= 2 * PI / 360;
                table[i].s.real = mag * cos(phase);
                table[i].s.imag = mag * sin(phase);
            }
        }
    }

    table = (struct data_pt *)STATIC_VAR(table);
    if (!table)
        return;
    if (ANALYSIS == MIF_AC) {
        double rv;

        size = PARAM_SIZE(table) / span;
        rv = RAD_FREQ;
        if (rv <= table[0].f) {
            ac_gain = table[0].s;
        } else if (rv >= table[size - 1].f) {
            ac_gain = table[size - 1].s;
        } else {
            for (i = 0; i < size; i++) {
                if (table[i].f > rv)
                    break;
            }

            /* Linear interpolation. */

            factor = (rv - table[i - 1].f) / (table[i].f - table[i - 1].f);
            ac_gain.real = table[i - 1].s.real +
                factor * (table[i].s.real - table[i - 1].s.real);
            ac_gain.imag = table[i - 1].s.imag +
                factor * (table[i].s.imag - table[i - 1].s.imag);
        }
        AC_GAIN(out, in) = ac_gain;
    } else { /* DC, transient ... */
        if (ANALYSIS == MIF_TRAN) {
            if (!STATIC_VAR(warned)) {
                STATIC_VAR(warned) = 1;
                cm_message_send("The xfer code model does not support "
                                "transient analysis.");
            }
        }
        OUTPUT(out) = table[0].s.real * INPUT(in);
    }
}


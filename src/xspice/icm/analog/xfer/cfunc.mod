/* Transfer function block for AC simulation, based on s_xfer code model. */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>

#define PI 3.141592653589793238462643383279502884197
#define ALLOC 1024

/* How the table information is stored internally. */

struct data {
    int            size; /* Table size. */
    double        *f;    /* Frequency table, radians/sec. */
    Mif_Complex_t *s;    /* The S-parameter table. */
};

static void cleanup(ARGS, Mif_Callback_Reason_t reason)
{
    struct data *table;

    switch (reason) {
    case MIF_CB_DESTROY:
        table = (struct data *)STATIC_VAR(table);
        if (table) {
            free(table->f);
            free(table->s);
            free(table);
            STATIC_VAR(table) = NULL;
        }
        break;
    }
}

static double *read_file(const char *fn, int span, int offset,
                         int *size_p, Mif_Boolean_t *ri, Mif_Boolean_t *db)
{
    FILE   *fp;
    double *file_data;
    double  mult;
    int     i, j, line, size, count, skip, want;
    char   *cp, *word;
    double  vals[9];
    char    buff[1024];

    fp = fopen_with_path(fn, "r");
    if (fp == NULL) {
        cm_message_printf("Can not open file %s: %s", fn, strerror(errno));
        return NULL;
    }

    /* Find and examine the option line. */

    line = 0;
    while (fgets(buff, sizeof buff, fp)) {
        ++line;
        if (buff[0] == '#')
            break;
    }
    if (buff[0] != '#') {
        cm_message_printf("No option line found in file %s", fn);
        fclose(fp);
        return NULL;
    }
    mult = 1.0e9; // Default to GHz
    for (cp = buff + 1; *cp; ++cp) {
        while (isspace(*cp))
            ++cp;
        if (*cp) {
            word = cp;
            while (*cp && !isspace(*cp))
                ++cp;
            *cp++ = '\0';

            if (!strcmp(word, "MHz"))
                mult = 1.0e6;
            else if (!strcmp(word, "KHz"))
                mult = 1.0e3;
            else if (!strcmp(word, "Hz"))
                mult = 1.0;
            else if (!strcmp(word, "DB"))
                *db = MIF_TRUE;
            else if (!strcmp(word, "RI"))
                *ri = MIF_TRUE;
        }
    }

    /* Read the data: at most 9 values per line. */

    size = ALLOC;
    file_data = malloc(size * sizeof(double));
    if (!file_data)
        goto bad;
    want = skip = i = 0;

    while (fgets(buff, sizeof buff, fp)) {
        ++line;
        count = sscanf(buff, "%lg%lg%lg%lg%lg%lg%lg%lg%lg",
                       vals, vals + 1, vals + 2, vals + 3, vals + 4,
                       vals + 5, vals + 6, vals + 7, vals + 8);
        if (!count)
            continue;
        if (span == 9 && count == 5) {
            /* Special case: noise data in 2-port file. */

            cm_message_printf("Ignoring noise parameters in file %s", fn);
            break;
        }
        if (skip) {
            if (count > skip) {
                j = skip;
                skip = 0;
            } else {
                skip -= count;
                continue;
            }
        } else {
            j = 0;
        }

        /* Check allocation. */

        if (i + 3 > size) {
            size += ALLOC;
            file_data = realloc(file_data, size * sizeof(double));
            if (!file_data)
                goto bad;
        }

        while (j < count) {
            /* Store value. */

            file_data[i++] = vals[j];
            switch (want) {
            case 0:
                /* Stored a frequency value. */

                if (j != 0)
                    cm_message_printf("Warning: frequency not at start "
                                      "of line %d",
                                      line);
                want = 2;
                j += offset;
                if (j >= count)
                    skip = j - count;
                break;
            case 1:
                want = 0;
                j += span - offset - 1;
                if (j >= count)
                    skip = j - count;
                break;
            case 2:
                j++;
                want = 1;
                skip = 0;
                break;
            }
        }
    }

    if (want || skip)
        cm_message_send("Warning: unexpected end of file data.");
    *size_p = i / 3;
 bad:
    fclose(fp);
    return file_data;
}

void cm_xfer(ARGS)  /* structure holding parms, inputs, outputs, etc.     */
{
    struct data    *table;
    Mif_Complex_t   ac_gain;
    double          factor;
    int             span, size, i;

    if (INIT) {
        Mif_Boolean_t  ri, db, rad;
        double        *file_data;
        int            offset, bad = 0, j;

        offset = PARAM(offset);
        span = PARAM(span);
        if (offset < 1 || span < offset + 2) {
            cm_message_send("Error: impossible span/offset.");
            return;
        }

        /* Table or File? */

        if (PARAM(file) == NULL) {
            /* Data given by "table" parameter. */

            file_data = NULL;
            size = PARAM_SIZE(table);
            bad = size % span || size == 0;

            /* Validate table. */

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
                cm_message_send("Error: badly formed table.");
                return;
            }
            size /= span;       /* Size now in entries. */
            ri = PARAM(r_i);
            db = PARAM(db);
            rad = PARAM(rad);
        } else {
            /* Get data from a Touchstone file. */

            if (!PARAM_NULL(table)) {
                cm_message_send("Error: both file and table "
                                "parameters given.");
                return;
            } else {
                db = MIF_FALSE;
                ri = MIF_FALSE;
                file_data = read_file(PARAM(file), span, offset,
                                      &size, &ri, &db);
                if (file_data == NULL)
                    return;
                span = 3;
                rad = MIF_FALSE;

                /* Explicit parameters override file. */

                if (!PARAM_NULL(r_i))
                    ri = PARAM(r_i);
                if (!PARAM_NULL(db))
                    db = PARAM(db);
                if (!PARAM_NULL(rad))
                    rad = PARAM(rad);
            }
        }

        /* Allocate the internal table. */

        table = (struct data *)malloc(sizeof(struct data));
        if (!table)
            return;
        table->size = size;
        table->f = (double*)malloc(size * sizeof (double));
        if (!table->f) {
            free(table);
            return;
        }
        table->s = ( Mif_Complex_t *)malloc(size * sizeof (Mif_Complex_t));
        if (!table->s) {
            free(table->f);
            free(table);
            return;
        }
        STATIC_VAR(table) = table;
        CALLBACK = cleanup;

        /* Fill it. */

        for (i = 0, j = 0; i < size; i++, j += span) {
            double f, a, b;

            if (file_data) {
                f = file_data[j];
                a = file_data[j + 1];
                b = file_data[j + 2];
            } else {
                f = PARAM(table[j]);
                a = PARAM(table[j + offset]);
                b = PARAM(table[j + offset + 1]);
            }
            table->f[i] = f * 2.0 * PI;
	    if (ri) {
                table->s[i].real = a;
                table->s[i].imag = b;
            } else {
                if (db)
                    a = pow(10, a / 20);
                if (!rad)
                    b *= 2 * PI / 360;
                table->s[i].real = a * cos(b);
                table->s[i].imag = a * sin(b);
            }
        }
        if (file_data)
            free(file_data);
    }
    table = (struct data *)STATIC_VAR(table);
    if (!table)
        return;
    if (ANALYSIS == MIF_AC) {
        double rv;

        size = table->size;
        rv = RAD_FREQ;
        if (rv <= table->f[0]) {
            ac_gain = table->s[0];
        } else if (rv >= table->f[size - 1]) {
            ac_gain = table->s[size - 1];
        } else {
            for (i = 0; i < size; i++) {
                if (table->f[i] > rv)
                    break;
            }

            /* Linear interpolation. */

            factor = (rv - table->f[i - 1]) / (table->f[i] - table->f[i - 1]);
            ac_gain.real = table->s[i - 1].real +
                factor * (table->s[i].real - table->s[i - 1].real);
            ac_gain.imag = table->s[i - 1].imag +
                factor * (table->s[i].imag - table->s[i - 1].imag);
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
        OUTPUT(out) = table->s[0].real * INPUT(in);
    }
}

/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE table2D/cfunc.mod

Copyright 2015
Holger Vogt

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA



AUTHORS

    03 Nov 2015     Holger Vogt


MODIFICATIONS


SUMMARY

    This file contains the model-specific routines used to
    functionally describe the 2D table code model used
    to read and interpolate a value from a 2D table from a file.

    The essentially non-oscillatory (ENO) interpolation in 2-D (eno2.c) is taken from the
    Madagascar Project at http://www.ahay.org/wiki/Main_Page
    Currently ENO is used only to obtain the derivatives,
    the data values are obtained by bilinear interpolation.
    This combination allows op convergence for some data tables (no guarantee though).

INTERFACES

    FILE                 ROUTINE CALLED

    N/A                  N/A


REFERENCED FILES

    Inputs from and outputs to ARGS structure.


NON-STANDARD FEATURES

    NONE

===============================================================================*/

/*=== INCLUDE FILES ====================*/

#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mada/eno2.h"

/*=== CONSTANTS ========================*/

#define OK 0
#define FAIL 1

/*=== MACROS ===========================*/

#if defined(__MINGW32__) || defined(_MSC_VER)
#define DIR_PATHSEP    "\\"
#else
#define DIR_PATHSEP    "/"
#endif

#if defined(_MSC_VER)
#define strdup _strdup
#define snprintf _snprintf
#endif

/*=== LOCAL VARIABLES & TYPEDEFS =======*/

struct filesource_state {
    FILE *fp;
    long pos;
    unsigned char atend;
};


typedef struct {

    int   ix;   /* size of array in x */
    int   iy;   /* size of array in y */

    struct filesource_state  *state;   /* the storage array for the
                                          filesource status. */
    int init_err;

    sf_eno2 newtable;   /* the table, code borrowed from madagascar project */

    double *xcol;   /* array of floats in x */
    double *ycol;   /* array of floats in y */

    double **table;

} Local_Data_t;

/* Type definition for each possible token returned. */
typedef enum token_type_s {CNV_NO_TOK, CNV_STRING_TOK} Cnv_Token_Type_t;

typedef char line_t[82]; /* A SPICE size line. <= 80 characters plus '\n\0' */

/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/

extern int findCrossOver(double arr[], int low, int high, double x);
extern double BilinearInterpolation(double q11, double q12, double q21, double q22, double x1, double x2, double y1, double y2, double x, double y);

extern char *CNVgettok(char **s);

/*==============================================================================

FUNCTION cnv_get_spice_value()

AUTHORS

    ???             Bill Kuhn

MODIFICATIONS

    30 Sep 1991     Jeffrey P. Murray

SUMMARY

    This function takes as input a string token from a SPICE
    deck and returns a floating point equivalent value.

INTERFACES

    FILE                 ROUTINE CALLED

    N/A                  N/A

RETURNED VALUE

    Returns the floating point value in pointer *p_value. Also
    returns an integer representing successful completion.

GLOBAL VARIABLES

    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== Static CNV_get_spice_value ROUTINE =============*/

/*
  Function takes as input a string token from a SPICE
  deck and returns a floating point equivalent value.
*/


static int
cnv_get_spice_value(char   *str,       /* IN - The value text e.g. 1.2K */
                    double *p_value)   /* OUT - The numerical value     */
{
    /* the following were "int4" devices - jpm */
    size_t  len;
    size_t  i;
    int     n_matched;

    line_t  val_str;

    char    c = ' ';
    char    c1;

    double  scale_factor;
    double  value;

    /* Scan the input string looking for an alpha character that is not  */
    /* 'e' or 'E'.  Such a character is assumed to be an engineering     */
    /* suffix as defined in the Spice 2G.6 user's manual.                */

    len = strlen(str);
    if (len > sizeof(val_str) - 1)
        len = sizeof(val_str) - 1;

    for (i = 0; i < len; i++) {
        c = str[i];
        if (isalpha(c) && (c != 'E') && (c != 'e'))
            break;
        else if (isspace(c))
            break;
        else
            val_str[i] = c;
    }
    val_str[i] = '\0';

    /* Determine the scale factor */

    if ((i >= len) || !isalpha(c))
        scale_factor = 1.0;
    else {

        if (isupper(c))
            c = (char) tolower(c);

        switch (c) {

        case 't':
            scale_factor = 1.0e12;
            break;

        case 'g':
            scale_factor = 1.0e9;
            break;

        case 'k':
            scale_factor = 1.0e3;
            break;

        case 'u':
            scale_factor = 1.0e-6;
            break;

        case 'n':
            scale_factor = 1.0e-9;
            break;

        case 'p':
            scale_factor = 1.0e-12;
            break;

        case 'f':
            scale_factor = 1.0e-15;
            break;

        case 'm':
            i++;
            if (i >= len) {
                scale_factor = 1.0e-3;
                break;
            }
            c1 = str[i];
            if (!isalpha(c1)) {
                scale_factor = 1.0e-3;
                break;
            }
            if (islower(c1))
                c1 = (char) toupper(c1);
            if (c1 == 'E')
                scale_factor = 1.0e6;
            else if (c1 == 'I')
                scale_factor = 25.4e-6;
            else
                scale_factor = 1.0e-3;
            break;

        default:
            scale_factor = 1.0;
        }
    }

    /* Convert the numeric portion to a float and multiply by the */
    /* scale factor.                                              */

    n_matched = sscanf(val_str, "%le", &value);

    if (n_matched < 1) {
        *p_value = 0.0;
        return FAIL;
    }

    *p_value = value * scale_factor;
    return OK;
}


/*==============================================================================

FUNCTION void cm_table2D()

AUTHORS

    08 Nov 2015    Holger Vogt

MODIFICATIONS



SUMMARY

    This function implements 2D table code model.

INTERFACES

    FILE                 ROUTINE CALLED

    N/A                  N/A


RETURNED VALUE

    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES

    NONE

NON-STANDARD FEATURES

    NONE

INPUT FILE SPEC

* Title (comments preceded by * ignored)
* table size
ix
iy
* x row independent variables
x0 x1 x2 x3 ... xix-1
y column independent variables
y0 y1 y2 y3 ... yiy-1
* table
x0y0 x1y0 x2y0 ... xix-1y0
...
x0yiy-1 x1yiy-1 x2yiy-1 ... xix-1yiy-1



==============================================================================*/


/*=== CM_table2D ROUTINE ===*/


void
cm_table2D(ARGS)   /* structure holding parms, inputs, outputs, etc. */
{
    int size, xind, yind;
    double xval, yval, xoff, yoff, xdiff, ydiff;
    double derivval[2], outval;
    double q11, q12, q21, q22, x1, x2, y1, y2;

    Local_Data_t *loc;   /* Pointer to local static data, not to be included
                            in the state vector */
    Mif_Complex_t ac_gain;

    size = PORT_SIZE(out);
    if (INIT == 1) {

        int i;
        int ix = 0,   /* elements in a row */
            iy = 0;   /* number of rows */
        double **table_data;
        double tmp;
        char *cFile, *cThisPtr, *cThisLine, *cThisLinePtr;
        int   isNewline;     /* Boolean indicating we've read a CR or LF */
        long  lFileLen;      /* Length of file */
        long  lIndex;        /* Index into cThisLine array */
        int   lLineCount;    /* Current line number */
        long  lStartPos;     /* Offset of start of current line */
        long  lTotalChars;   /* Total characters read */
        int   interporder;   /* order of interpolation for eno */

        /* allocate static storage for *loc */
        STATIC_VAR (locdata) = calloc(1, sizeof(Local_Data_t));
        loc = STATIC_VAR (locdata);

        /* Allocate storage for internal state */
        loc->state = (struct filesource_state*) malloc(sizeof(struct filesource_state));
        loc->ix = loc->iy = 0;

        /* open file */
        loc->state->fp = fopen_with_path(PARAM(file), "r");
        loc->state->pos = 0;
        loc->state->atend = 0;
        if (!loc->state->fp) {
            char *lbuffer, *p;
            lbuffer = getenv("NGSPICE_INPUT_DIR");
            if (lbuffer && *lbuffer) {
                p = (char*) malloc(strlen(lbuffer) + strlen(DIR_PATHSEP) + strlen(PARAM(file)) + 1);
                sprintf(p, "%s%s%s", lbuffer, DIR_PATHSEP, PARAM(file));
                loc->state->fp = fopen(p, "r");
                free(p);
            }
            if (!loc->state->fp) {
                cm_message_printf("cannot open file %s", PARAM(file));
                loc->state->atend = 1;
                loc->init_err = 1;
                return;
            }
        }
        /* get file length */
        fseek(loc->state->fp, 0L, SEEK_END);   /* Position to end of file */
        lFileLen = ftell(loc->state->fp);      /* Get file length */
        rewind(loc->state->fp);                /* Back to start of file */

        /* create string to hold the whole file */
        cFile = calloc(lFileLen + 1, sizeof(char));
        /* create another string long enough for file manipulation */
        cThisLine = calloc(lFileLen + 1, sizeof(char));
        if (cFile == NULL || cThisLine == NULL) {
            cm_message_printf("Insufficient memory to read file %s", PARAM(file));
            loc->state->atend = 1;
            loc->init_err = 1;
            return;
        }
        /* read whole file into memory */
        fread(cFile, lFileLen, 1, loc->state->fp); /* Read the entire file into cFile */
        fclose(loc->state->fp);

        cThisPtr = cFile;
        cThisLinePtr = cThisLine;
        lLineCount  = 0L;
        lTotalChars = 0L;

        while (*cThisPtr) {               /* Read until reaching null char */

            lIndex    = 0L;               /* Reset counters and flags */
            isNewline = 0;
            lStartPos = lTotalChars;

            while (*cThisPtr) {           /* Read until reaching null char */

                if (!isNewline) {         /* Haven't read a CR or LF yet */
                    if (*cThisPtr == '\r' || *cThisPtr == '\n') /* This char IS a CR or LF */
                        isNewline = 1;    /* Set flag */
                }

                else if (*cThisPtr != '\r' && *cThisPtr != '\n') /* Already found CR or LF */
                    break;                /* Done with line */

                cThisLinePtr[lIndex++] = *cThisPtr++; /* Add char to output and increment */
                lTotalChars++;
            }

            cThisLinePtr[lIndex] = '\0';       /* Terminate the string */
            lLineCount++;                      /* Increment the line counter */
            /* continue if comment or empty */
            if (cThisLinePtr[0] == '*' || cThisLinePtr[0] == '\0') {
                lLineCount--;   /* we count only real lines */
                continue;
            }

            if (lLineCount == 1) {
                cnv_get_spice_value(cThisLinePtr, &tmp);
                loc->ix = ix = (int) tmp;
                /* generate row  data structure (x) */
                loc->xcol = (double*) calloc(ix, sizeof(double));
            }
            else if (lLineCount == 2) {
                cnv_get_spice_value(cThisLinePtr, &tmp);
                loc->iy = iy = (int) tmp;
                /* generate  column data structure (y) */
                loc->ycol = (double*) calloc(iy, sizeof(double));
            }
            else if (lLineCount == 3) {
                char *token = CNVgettok(&cThisLinePtr);
                i = 0;
                while (token) {
                    if (i == ix) {
                        cm_message_printf("Too many numbers in x row.");
                        loc->init_err = 1;
                        return;
                    }
                    cnv_get_spice_value(token, &loc->xcol[i++]);
                    free(token);
                    token = CNVgettok(&cThisLinePtr);
                }
                if (i < ix) {
                    cm_message_printf("Not enough numbers in x row.");
                    loc->init_err = 1;
                    return;
                }
            }
            else if (lLineCount == 4) {
                char *token = CNVgettok(&cThisLinePtr);
                i = 0;
                while (token) {
                    if (i == iy) {
                        cm_message_printf("Too many numbers in y row.");
                        loc->init_err = 1;
                        return;
                    }
                    cnv_get_spice_value(token, &loc->ycol[i++]);
                    free(token);
                    token = CNVgettok(&cThisLinePtr);
                }
                if (i < iy) {
                    cm_message_printf("Not enough numbers in y row.");
                    loc->init_err = 1;
                    return;
                }
                /* jump out of while loop to read in the table */
                break;
            }
        }

        /* generate table core */
        interporder = PARAM(order);
        /* boundary limits set to param 'order' aren't recognized,
           so limit them here */
        if (interporder < 2) {
            cm_message_printf("Parameter Order=%d not possible, set to minimum value 2", interporder);
            interporder = 2;
        }
        /* int order : interpolation order,
           int n1, int n2 : data dimensions */
        loc->newtable = sf_eno2_init(interporder, ix, iy);

        /* create table_data in memory */
        /* data [n2][n1] */
        table_data = calloc(iy, sizeof(double *));
        for (i = 0; i < iy; i++)
            table_data[i] = calloc(ix, sizeof(double));

        loc->table = table_data;

        /* continue reading from cFile */
        lLineCount = 0;
        while (*cThisPtr) {               /* Read until reaching null char */
            char *token;

            lIndex    = 0L;               /* Reset counters and flags */
            isNewline = 0;
            lStartPos = lTotalChars;

            while (*cThisPtr) {           /* Read until reaching null char */

                if (!isNewline) {         /* Haven't read a CR or LF yet */
                    if (*cThisPtr == '\r' || *cThisPtr == '\n') /* This char IS a CR or LF */
                        isNewline = 1;    /* Set flag */
                }

                else if (*cThisPtr != '\r' && *cThisPtr != '\n') /* Already found CR or LF */
                    break;                /* Done with line */

                cThisLinePtr[lIndex++] = *cThisPtr++; /* Add char to output and increment */
                lTotalChars++;
            }

            cThisLinePtr[lIndex] = '\0';       /* Terminate the string */
            lLineCount++;                      /* Increment the line counter */
            /* continue if comment or empty */
            if (cThisLinePtr[0] == '*' || cThisLinePtr[0] == '\0') {
                if (lTotalChars >= lFileLen) {
                    cm_message_printf("Not enough data in file %s", PARAM(file));
                    loc->init_err = 1;
                    return;
                }
                lLineCount--;   /* we count only real lines */
                continue;
            }
            token = CNVgettok(&cThisLinePtr);
            i = 0;
            while (token) {
                double tmpval;
                if (i == ix) {
                    cm_message_printf("Too many numbers in y row no. %d.", lLineCount);
                    loc->init_err = 1;
                    return;
                }

                /* read table core from cFile, fill local static table structure table_data */
                cnv_get_spice_value(token, &tmpval);
                table_data[lLineCount - 1][i++] = (float) tmpval; /* fixme (float) */
                free(token);
                token = CNVgettok(&cThisLinePtr);
            }
            if (i < ix) {
                cm_message_printf("Not enough numbers in y row no. %d.", lLineCount);
                loc->init_err = 1;
                return;
            }
        }

        /* fill table data into eno2 structure */
        sf_eno2_set (loc->newtable, table_data /* data [n2][n1] */);

        /* free all the emory allocated */
        // for (i = 0; i < iy; i++)
        //     free(table_data[i]);
        // free(table_data);
        free(cFile);
        free(cThisLine);
    } /* end of initialization "if (INIT == 1)" */

    loc = STATIC_VAR (locdata);

    /* return immediately if there was an initialization error */
    if (loc->init_err == 1)
        return;

    /* get input x, y,
       find corresponding indices,
       get x and y offsets,
       call interpolation function with value and derivative */
    xval = INPUT(inx);
    yval = INPUT(iny);

    /* find index */
    if (xval < loc->xcol[0] || xval > loc->xcol[loc->ix - 1]) {
        if (PARAM(verbose) > 0)
            cm_message_printf("x value %g exceeds table limits,\n"
                              "  please enlarge range of your table",
                              xval);
        return;
    }
    if (yval < loc->ycol[0] || yval > loc->ycol[loc->iy - 1]) {
        if (PARAM(verbose) > 0)
            cm_message_printf("y value %g exceeds table limits,\n"
                              "  please enlarge range of your table",
                              yval);
        return;
    }

    /* something like binary search to get the index */
    xind = findCrossOver(loc->xcol, 0, loc->ix - 1, xval);
    /* find index with minimum distance between xval and row value */
    if (fabs(loc->xcol[xind + 1] - xval) < fabs(xval - loc->xcol[xind]))
        xind++;
    xoff = xval - loc->xcol[xind];
    yind = findCrossOver(loc->ycol, 0, loc->iy - 1, yval);
    /* find index with minimum distance between yval and column value */
    if (fabs(loc->ycol[yind + 1] - yval) < fabs(yval - loc->ycol[yind]))
        yind++;
    yoff = yval - loc->ycol[yind];

    /* find local difference around index of independent row and column values */
    if (xind == loc->ix - 1)
        xdiff = loc->xcol[xind] - loc->xcol[xind - 1];
    else if (xind == 0)
        xdiff = loc->xcol[xind + 1] - loc->xcol[xind];
    else
        xdiff = 0.5 * (loc->xcol[xind + 1] - loc->xcol[xind - 1]);

    if (yind == loc->iy - 1)
        ydiff = loc->ycol[yind] - loc->ycol[yind - 1];
    else if (yind == 0)
        ydiff = loc->ycol[yind + 1] - loc->ycol[yind];
    else
        ydiff = 0.5 * (loc->ycol[yind + 1] - loc->ycol[yind - 1]);

    /* Essentially non-oscillatory (ENO) interpolation to obtain the derivatives only.
       Using outval for now yields ngspice op non-convergence */
    sf_eno2_apply (loc->newtable,
                   xind, yind,   /* grid location */
                   xoff, yoff,   /* offset from grid */
                   &outval,      /* output data value */
                   derivval,     /* output derivatives [2] */
                   DER           /* what to compute [FUNC, DER, BOTH] */
                   );

    /* bilinear interpolation to obtain the output value */
    xind = findCrossOver(loc->xcol, 0, loc->ix - 1, xval);
    yind = findCrossOver(loc->ycol, 0, loc->iy - 1, yval);
    x1 = loc->xcol[xind];
    x2 = loc->xcol[xind + 1];
    y1 = loc->ycol[yind];
    y2 = loc->ycol[yind + 1];
    q11 = loc->table[yind][xind];
    q12 = loc->table[yind + 1][xind];
    q21 = loc->table[yind][xind + 1];
    q22 = loc->table[yind + 1][xind + 1];
    outval = BilinearInterpolation(q11, q12, q21, q22, x1, x2, y1, y2, xval, yval);

    if (ANALYSIS != MIF_AC) {
        double xderiv, yderiv, outv;
        outv = PARAM(offset) + PARAM(gain) * outval;
        OUTPUT(out) = outv;
        xderiv = PARAM(gain) * derivval[0] / xdiff;
        PARTIAL(out, inx) = xderiv;
        yderiv = PARAM(gain) * derivval[1] / ydiff;
        PARTIAL(out, iny) = yderiv;

        if (PARAM(verbose) > 1)
            cm_message_printf("\nI: %g, xval: %g, yval: %g, xderiv: %g, yderiv: %g", outv, xval, yval, xderiv, yderiv);
    }
    else {
        ac_gain.real = PARAM(gain) * derivval[0] / xdiff;
        ac_gain.imag= 0.0;
        AC_GAIN(out, inx) = ac_gain;
        ac_gain.real = PARAM(gain) * derivval[1] / ydiff;
        ac_gain.imag= 0.0;
        AC_GAIN(out, iny) = ac_gain;
    }

}

/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE table2D/cfunc.mod

-------------------------------------------------------------------------
 Copyright 2015
 The ngspice team
 All Rights Reserved
 GPL
 (see COPYING or https://opensource.org/licenses/GPL-2.0)
-------------------------------------------------------------------------

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

    10 Aug 2018     Holger Vogt

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
#include <stdbool.h>
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <sys/types.h>
#include <sys/stat.h>

#include "mada/eno2.h"

typedef struct {
    int   ix;   /* size of array in x */
    int   iy;   /* size of array in y */

    sf_eno2 newtable;   /* the table, code borrowed from madagascar project */

    /* Input values corresponding to each index. They define the value
     * in the domain at each index value */
    double *xcol;   /* array of doubles in x */
    double *ycol;   /* array of doubles in y */

    double **table; /* f(xi, yj) */
} Table2_Data_t;

typedef Table2_Data_t Local_Data_t;

/*=== MACROS ===========================*/

#if defined(__MINGW32__) || defined(_MSC_VER)
#define DIR_PATHSEP    "\\"
#else
#define DIR_PATHSEP    "/"
#endif

#if defined(_MSC_VER)
#define strdup _strdup
#endif

/*=== LOCAL VARIABLES & TYPEDEFS =======*/

struct filesource_state {
    FILE *fp;
    long pos;
    unsigned char atend;
};





/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/

double BilinearInterpolation(double x, double y,
        int xind, int yind, double **td);
extern char *CNVgettok(char **s);
int cnv_get_spice_value(char *str, double *p_value);
extern int findCrossOver(double arr[], int n, double x);

static void free_local_data(Table2_Data_t *loc);
static inline double get_local_diff(int n, double *col, int ind);
static Table2_Data_t *init_local_data(const char *filename, int order);



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




static void cm_table2D_callback(ARGS,
        Mif_Callback_Reason_t reason)
{
    switch (reason) {
        case MIF_CB_DESTROY: {
            Table2_Data_t *loc = STATIC_VAR(locdata);
            if (loc) {
                free_local_data(loc);
                STATIC_VAR(locdata) = loc = NULL;
            }
            break;
        } /* end of case MIF_CB_DESTROY */
    } /* end of switch over reason being called */
} /* end of function cm_table2D_callback */



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
* y column independent variables
y0 y1 y2 y3 ... yiy-1
* table
x0y0 x1y0 x2y0 ... xix-1y0
...
x0yiy-1 x1yiy-1 x2yiy-1 ... xix-1yiy-1



==============================================================================*/

/* How quickly partial derivative ramps to zero at boundary. */

#define RAMP_WIDTH 0.125

/*=== CM_table2D ROUTINE ===*/

static const char exceed_fmt[] = "%c value %g exceeds table limits,\n"
                                 "  please enlarge range of your table.";

void cm_table2D(ARGS)   /* structure holding parms, inputs, outputs, etc. */
{
    int size, xind, yind;
    double xval, yval, xoff, yoff, xdiff, ydiff, xramp, yramp;
    double derivval[2], outval;

    Table2_Data_t *loc;   /* Pointer to local static data, not to be included
                            in the state vector */

    size = PORT_SIZE(out);
    if (INIT == 1) { /* Must do initializations */
        STATIC_VAR(locdata) = init_local_data(
                PARAM(file),
                PARAM(order));
        CALLBACK = cm_table2D_callback;
    }

    /* return immediately if there was an initialization error */
    if ((loc = STATIC_VAR(locdata)) == (Table2_Data_t *) NULL) {
        return;
    }

    /* get input x, y;
       find corresponding indices;
       get x and y offsets;
       call interpolation functions with value and derivative */
    xval = INPUT(inx);
    yval = INPUT(iny);

    /* check table ranges */
    if (xval < loc->xcol[0] || xval > loc->xcol[loc->ix - 1]) {
        /* x input out of region: use nearest point in region, ramping
         * partial derivative to zero at edge.
         */

        if (PARAM(verbose) > 0)
            cm_message_printf(exceed_fmt, 'x', xval);
	if (xval < loc->xcol[0]) {
            xramp = 1 - ((loc->xcol[0] - xval) /
                         (RAMP_WIDTH * (loc->xcol[1] - loc->xcol[0])));
            if (xramp < 0.0)
                xramp = 0.0;
            xval = loc->xcol[0];
        } else {
            xramp = 1 - ((xval - loc->xcol[loc->ix - 1]) /
                         (RAMP_WIDTH * (loc->xcol[loc->ix - 1] -
                                        loc->xcol[loc->ix - 2])));
            if (xramp < 0.0)
                xramp = 0.0;
            xval = loc->xcol[loc->ix - 1];
        }
    } else {
        xramp = 1.0;
    }
    if (yval < loc->ycol[0] || yval > loc->ycol[loc->iy - 1]) {
        if (PARAM(verbose) > 0)
            cm_message_printf(exceed_fmt, 'y', yval);
	if (yval < loc->ycol[0]) {
            yramp = 1 - ((loc->ycol[0] - yval) /
                         (RAMP_WIDTH * (loc->ycol[1] - loc->ycol[0])));
            if (yramp < 0.0)
                yramp = 0.0;
            yval = loc->ycol[0];
        } else {
            yramp = 1 - ((yval - loc->ycol[loc->iy - 1]) /
                         (RAMP_WIDTH * (loc->ycol[loc->iy - 1] -
                                        loc->ycol[loc->iy - 2])));
            if (yramp < 0.0)
                yramp = 0.0;
            yval = loc->ycol[loc->iy - 1];
        }
    } else {
        yramp = 1.0;
    }

    /*** find indices where interpolation will be done ***/
    /* something like binary search to get the index */
    xind = findCrossOver(loc->xcol, loc->ix, xval);
    xoff = xval - loc->xcol[xind];
    yind = findCrossOver(loc->ycol, loc->iy, yval);
    yoff = yval - loc->ycol[yind];

    /* Find local difference around index of independent row and
     * column values */
    xdiff = get_local_diff(loc->ix, loc->xcol, xind);
    ydiff = get_local_diff(loc->iy, loc->ycol, yind);

    /* Essentially non-oscillatory (ENO) interpolation to obtain the derivatives only.
       Using outval for now yields ngspice op non-convergence */
    sf_eno2_apply(loc->newtable,
                  xind, yind,   /* grid location */
                  xoff, yoff,   /* offset from grid */
                  &outval,      /* output data value */
                  derivval,     /* output derivatives [2] */
                  DER           /* what to compute [FUNC, DER, BOTH] */
                  );

    /* xind may become too large when xval == xcol[loc->ix - 1] */
    if (xind == loc->ix - 1) {
        --xind;
	xoff += loc->xcol[xind + 1] - loc->xcol[xind];
    }
    if (yind == loc->iy - 1) {
        --yind;
	yoff += loc->ycol[yind + 1] - loc->ycol[yind];
    }

    /* Overwrite outval from sf_eno2_apply by bilinear interpolation */
    outval = BilinearInterpolation(
            xoff / (loc->xcol[xind + 1] - loc->xcol[xind]),
            yoff / (loc->ycol[yind + 1] - loc->ycol[yind]),
            xind, yind, loc->table);

    if (ANALYSIS != MIF_AC) {
        double xderiv, yderiv, outv;
        outv = PARAM(offset) + PARAM(gain) * outval;
        OUTPUT(out) = outv;
        xderiv = xramp * PARAM(gain) * derivval[0] / xdiff;
        PARTIAL(out, inx) = xderiv;
        yderiv = yramp * PARAM(gain) * derivval[1] / ydiff;
        PARTIAL(out, iny) = yderiv;

        if (PARAM(verbose) > 1) {
            cm_message_printf("\nI: %g, xval: %g, yval: %g, "
                    "xderiv: %g, yderiv: %g",
                    outv, xval, yval, xderiv, yderiv);
        }
    }
    else {
        Mif_Complex_t ac_gain;
        ac_gain.real = xramp * PARAM(gain) * derivval[0] / xdiff;
        ac_gain.imag= 0.0;
        AC_GAIN(out, inx) = ac_gain;
        ac_gain.real = yramp * PARAM(gain) * derivval[1] / ydiff;
        ac_gain.imag= 0.0;
        AC_GAIN(out, iny) = ac_gain;
    }
} /* end of function cm_table2D */



/* This function initializes local data */
static Table2_Data_t *init_local_data(const char *filename, int order)
{
    int xrc = 0;
    int ix = 0,   /* elements in a row */
        iy = 0;   /* number of rows */

    double **table_data;
    double tmp;
    FILE *fp = (FILE *) NULL; /* Handle to file */
    char *cFile = (char *) NULL;
    char *cThisLine = (char *) NULL;
    char *cThisPtr, *cThisLinePtr;
    size_t lFileLen;     /* Length of file */
    size_t lFileRead;    /* Length of file read in */
    int   lLineCount;    /* Current line number */
    size_t lTotalChar;   /* Total characters read */
    int   interporder;   /* order of interpolation for eno */
    Table2_Data_t *loc = (Table2_Data_t *) NULL; /* local data */


    /* Allocate static storage for *loc */
    if ((loc = (Table2_Data_t *) calloc(1,
            sizeof(Table2_Data_t))) == (Table2_Data_t *) NULL) {
        cm_message_printf("cannot allocate memory for lookup table.");
        xrc = -1;
        goto EXITPOINT;
    }

    /* Init row and column counts to 0 (actually already were due
     * to calloc) */
    loc->ix = loc->iy = 0;

    /* open file */
    fp = fopen_with_path(filename, "r");
    if (!fp) { /* Standard open attempt failed */
        const char * const lbuffer = getenv("NGSPICE_INPUT_DIR");
        if (lbuffer && *lbuffer) {
            char * const p = (char *) malloc(strlen(lbuffer) +
                    strlen(DIR_PATHSEP) +
                    strlen(filename) + 1);
            if (p == (char *) NULL) {
                cm_message_printf("cannot allocate buffer to "
                        "attempt alternate file open.");
                xrc = -1;
                goto EXITPOINT;
            }
            (void) sprintf(p, "%s%s%s",
                    lbuffer, DIR_PATHSEP, filename);
            fp = fopen(p, "r");
            free(p);
        }
    }

    /* Test for valid file pointer */
    if (!fp) {
        cm_message_printf("cannot open file %s", filename);
        xrc = -1;
        goto EXITPOINT;
    }

    /* Find the size of the data file */
    {
        struct stat st;
        if (fstat(fileno(fp), &st)) {
            cm_message_printf("cannot get length of file %s",
                    filename);
            xrc = -1;
            goto EXITPOINT;
        }
        /* Copy file length */
        lFileLen = (size_t) st.st_size;
    }

    /* create string to hold the whole file */
    cFile = calloc(lFileLen + 1, sizeof(char));
    /* create another string long enough for file manipulation */
    cThisLine = calloc(lFileLen + 1, sizeof(char));
    if (cFile == NULL || cThisLine == NULL) {
        cm_message_printf("Insufficient memory to read file %s",
                filename);
        xrc = -1;
        goto EXITPOINT;
    }

    /* read whole file into cFile */
    {
        /* Number of chars read may be less than lFileLen, because /r are
         * skipped by 'fread' when file opened in text mode */
        lFileRead = fread(cFile, sizeof(char), lFileLen, fp);
        const int file_error = ferror(fp);
        fclose(fp); /* done with file */
        fp = (FILE *) NULL;
        if (file_error) {
            cm_message_printf("Error reading data file %s", filename);
            xrc = -1;
            goto EXITPOINT;
        }
    }
    /* Number of chars read may be less than lFileLen, because /r are
     * skipped by 'fread' when file opened in text mode */
    cFile[lFileRead] = '\0';

    cThisPtr = cFile;
    cThisLinePtr = cThisLine;
    lLineCount  = 0L;
    lTotalChar = 0L;

    while (*cThisPtr) { /* Read until reaching null char */
        long lIndex = 0L; /* Index into cThisLine array */
        bool isNewline = false; /* Boolean indicating read a CR or LF */

        while (*cThisPtr) { /* Read until reaching null char */
            if (!isNewline) { /* Haven't read a LF yet */
                if (*cThisPtr == '\n') { /* This char is a LF */
                    isNewline = true;    /* Set flag */
                }
            }
            else if (*cThisPtr != '\n') { /* Already found LF */
                break;                /* Done with line */
            }

            /* Add char to output and increment */
            cThisLinePtr[lIndex++] = *cThisPtr++;
            lTotalChar++;
        }

        cThisLinePtr[lIndex] = '\0';       /* Terminate the string */
        lLineCount++;                      /* Increment the line counter */
        /* continue if comment or empty */
        if (cThisLinePtr[0] == '*' || cThisLinePtr[0] == '\n') {
            lLineCount--;   /* we count only real lines */
            continue;
        }

        if (lLineCount == 1) {
            cnv_get_spice_value(cThisLinePtr, &tmp);
            loc->ix = ix = (int) tmp;
            /* generate row  data structure (x) */
            if ((loc->xcol = (double *) calloc((size_t) ix,
                    sizeof(double))) == (double *) NULL) {
                cm_message_printf("Unable to allocate row structure.");
                xrc = -1;
                goto EXITPOINT;
            }
        }
        else if (lLineCount == 2) {
            cnv_get_spice_value(cThisLinePtr, &tmp);
            loc->iy = iy = (int) tmp;
            /* generate  column data structure (y) */
            if ((loc->ycol = (double *) calloc((size_t) iy,
                    sizeof(double))) == (double *) NULL) {
                cm_message_printf("Unable to allocate column structure.");
                xrc = -1;
                goto EXITPOINT;
            }
        }
        else if (lLineCount == 3) {
            char *token = CNVgettok(&cThisLinePtr);
            int i = 0;
            while (token) {
                if (i == ix) {
                    cm_message_printf("Too many numbers in x row.");
                    free(token);
                    xrc = -1;
                    goto EXITPOINT;
                }
                cnv_get_spice_value(token, &loc->xcol[i++]);
                free(token);
                token = CNVgettok(&cThisLinePtr);
            }
            if (i < ix) {
                cm_message_printf("Not enough numbers in x row.");
                xrc = -1;
                goto EXITPOINT;
            }
        }
        else if (lLineCount == 4) {
            char *token = CNVgettok(&cThisLinePtr);
            int i = 0;
            while (token) {
                if (i == iy) {
                    cm_message_printf("Too many numbers in y row.");
                    xrc = -1;
                    goto EXITPOINT;
                }
                cnv_get_spice_value(token, &loc->ycol[i++]);
                free(token);
                token = CNVgettok(&cThisLinePtr);
            }
            if (i < iy) {
                cm_message_printf("Not enough numbers in y row.");
                xrc = -1;
                goto EXITPOINT;

            }
            /* jump out of while loop to read in the table */
            break;
        }
    }

    /* generate table core */
    interporder = order;
    /* boundary limits set to param 'order' aren't recognized,
       so limit them here */
    if (interporder < 2) {
        cm_message_printf("Parameter Order=%d not possible, "
                "set to minimum value 2",
                interporder);
        interporder = 2;
    }
    /* int order : interpolation order,
       int n1, int n2 : data dimensions */
    if ((loc->newtable = sf_eno2_init(
            interporder, ix, iy)) == (sf_eno2) NULL) {
        cm_message_printf("eno2 initialization failure.");
        xrc = -1;
        goto EXITPOINT;

    }

    /* create table_data in memory */
    /* data [n2][n1] */
    if ((loc->table = table_data = (double **) calloc((size_t) iy,
            sizeof(double *))) == (double **) NULL) {
        cm_message_printf("Unable to allocate data table.");
        free(cFile);
        free(cThisLine);
        free_local_data(loc);
        return (Local_Data_t *) NULL;
    }

    {
        int i;
        for (i = 0; i < iy; i++) {
            if ((table_data[i] = (double *) calloc((size_t) ix,
                    sizeof(double))) == (double *) NULL) {
                cm_message_printf("Unable to allocate data table "
                        "row %d", i + 1);
                free(cFile);
                free(cThisLine);
                free_local_data(loc);
                return (Local_Data_t *) NULL;
            }
        }
    }

    loc->table = table_data; /* give to local data structure */


    /* continue reading f(x,y) values from cFile */
    lLineCount = 0;
    while (*cThisPtr) {           /* Read until reaching null char */
            char *token;
            long int lIndex = 0;    /* Index into cThisLine array */
            bool isNewline = 0;

        while (*cThisPtr) {       /* Read until reaching null char */

            if (!isNewline) {         /* Haven't read a LF yet */
                if (*cThisPtr == '\n') /* This char is a LF */
                    isNewline = 1;    /* Set flag */
            }

            else if (*cThisPtr != '\n') { /* Already found LF */
                break;                /* Done with line */
            }

            /* Add char to output and increment */
            cThisLinePtr[lIndex++] = *cThisPtr++;
            lTotalChar++;
        }

        cThisLinePtr[lIndex] = '\0'; /* Terminate the string */
        lLineCount++; /* Increment the line counter */
        /* continue if comment or empty */
        if (cThisLinePtr[0] == '*' || cThisLinePtr[0] == '\0') {
            if (lTotalChar >= lFileLen) {
                cm_message_printf("Not enough data in file %s",
                        filename);
                free(cFile);
                free(cThisLine);
                free_local_data(loc);
                return (Local_Data_t *) NULL;
            }
            lLineCount--;   /* we count only real lines */
            continue;
        }
        token = CNVgettok(&cThisLinePtr);
        {
            int i = 0;
            while (token) {
                double tmpval;
                if (i == ix) {
                    cm_message_printf("Too many numbers in y row no. %d.",
                            lLineCount);
                    xrc = -1;
                    goto EXITPOINT;
                }

                /* read table core from cFile, fill local static table
                 * structure table_data */
                cnv_get_spice_value(token, &tmpval);
                table_data[lLineCount - 1][i++] = tmpval;
                free(token);
                token = CNVgettok(&cThisLinePtr);
            }
            if (i < ix) {
                cm_message_printf("Not enough numbers in y row no. %d.",
                        lLineCount);
                xrc = -1;
                goto EXITPOINT;
            }
        }
    } /* end of loop over characters read from file */

    /* fill table data into eno2 structure */
    sf_eno2_set(loc->newtable, table_data /* data [n2][n1] */);

EXITPOINT:
    /* free the file and memory allocated */
    if (cFile != (char *) NULL) {
        free(cFile);
    }
    if (cThisLine != (char *) NULL) {
        free(cThisLine);
    }
    if (fp != (FILE *) NULL) {
        (void) fclose(fp);
    }

    /* On error free any initialization that was started */
    if (xrc != 0) {
        if (loc != (Table2_Data_t *) NULL) {
            free_local_data(loc);
            loc = (Table2_Data_t *) NULL;
        }
    }
    return loc;
} /* end of function init_local_data */



/* Free memory allocations in Local_Data_t structure */
static void free_local_data(Table2_Data_t *loc)
{
    if (loc == (Table2_Data_t *) NULL) {
        return;
    }

    /* Free data table and related values */
    if (loc->table) {
        int i;
        int n_y = loc->iy;
        for (i = 0; i < n_y; i++) {
            free(loc->table[i]);
        }

        free(loc->table);
    }

    free(loc->xcol);
    free(loc->ycol);
    sf_eno2_close(loc->newtable);
    free(loc);
} /* end of function free_local_data */



/* Finds difference between column values */
static inline double get_local_diff(int n, double *col, int ind)
{
    if (ind >= n - 1) {
        return col[n - 1] - col[n - 2];
    }
    if (ind <= 0) {
        return col[1] - col[0];
    }
    return 0.5 * (col[ind + 1] - col[ind - 1]);
} /* end of function get_local_diff */




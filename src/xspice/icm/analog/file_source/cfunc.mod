/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE filesource/cfunc.mod

Copyright 2011
Thomas Sailer

              

AUTHORS                      

    20 May 2011     Thomas Sailer
    03 Sep 2012     Holger Vogt
    27 Feb 2017     Marcel Hendrix
    23 JUL 2018     Holger Vogt


MODIFICATIONS


SUMMARY

    This file contains the model-specific routines used to
    functionally describe the file source code model used
    to read an array from a file containing lines with
    time and analog values, and returning them per time step.


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

/*=== CONSTANTS ========================*/




/*=== MACROS ===========================*/

#if defined(__MINGW32__) || defined(_MSC_VER)
#define DIR_PATHSEP    "\\"
#else
#define DIR_PATHSEP    "/"
#endif

/* For WIN32, make  strdup become _strdup unless it is defined already,
 * as it would be if CRT debugging is being used */
#if defined(_WIN32) && !defined(strdup)
#define strdup _strdup
#endif

/*=== LOCAL VARIABLES & TYPEDEFS =======*/                         

struct filesource_state {
    FILE *fp;
    unsigned char atend;
};


struct infiledata {
    double *datavec;
    size_t vecallocated;
    int maxoccupied;
    int actpointer;
    int size;
};

typedef struct {

    double   *amplinterval;   /* the storage array for the
                                   amplitude offsets   */

    double   *timeinterval;   /* the storage array for the
                                   time offset   */

    struct filesource_state  *state;   /* the storage array for the
                                          filesource status.    */

    struct infiledata *indata; /* the storage vector for the input data
                                  sourced from file. */

} Local_Data_t;

           
/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/




                   
/*==============================================================================

FUNCTION void cm_filesource()

AUTHORS                      

    20 May 2011     Thomas Sailer

MODIFICATIONS   

    07 Sept 2012    Holger Vogt
    27 Feb  2017    Marcel Hendrix
    23 JUL  2018    Holger Vogt

SUMMARY

    This function implements the filesource code model.

INTERFACES       

    FILE                 ROUTINE CALLED     

    N/A                  N/A


RETURNED VALUE
    
    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/


/*=== CM_FILESOURCE ROUTINE ===*/

static void cm_filesource_callback(ARGS, Mif_Callback_Reason_t reason);

void cm_filesource(ARGS)   /* structure holding parms, inputs, outputs, etc.     */
{
    int size = PORT_SIZE(out);
    int stepsize = size + 1;
    int amplscalesize;
    int amploffssize;
    int j;

    Local_Data_t *loc;        /* Pointer to local static data, not to be included
                                       in the state vector */

    if(ANALYSIS == MIF_AC) {
        return;
    }
    if (INIT == 1) {

        int count;

        /*** allocate static storage for *loc ***/
        if ((loc = (Local_Data_t *) (STATIC_VAR(locdata) = calloc(1,
                sizeof(Local_Data_t)))) == (Local_Data_t *) NULL) {
            cm_message_send("Unable to allocate Local_Data_t "
                    "in cm_filesource()");
            return;
        }

        /* Allocate storage for internal state */
        loc->timeinterval = (double *) calloc(2, sizeof(double));
        loc->amplinterval = (double *) calloc(2 * (size_t) size,
                sizeof(double));
        loc->state = (struct filesource_state *) calloc(1,
                sizeof(struct filesource_state)); /* calloc to null fp */
        loc->indata = (struct infiledata *) malloc(
                sizeof(struct infiledata));
        loc->indata->datavec = (double *) malloc(sizeof(double) *
                (size_t) (stepsize * 1000));

        /* Check allocations */
        if (loc->timeinterval == (double *) NULL ||
                loc->amplinterval == (double *) NULL ||
                loc->state == (struct filesource_state *) NULL ||
                loc->indata == (struct infiledata *) NULL ||
                loc->indata->datavec == (double *) NULL) {
            cm_message_send("Unable to allocate Local_Data_t  fields "
                    "in cm_filesource()");
            cm_filesource_callback(mif_private, MIF_CB_DESTROY);
            return;
        }

        CALLBACK = cm_filesource_callback;

        loc->indata->vecallocated = (size_t) (stepsize * 1000);
        loc->indata->maxoccupied = 0;
        loc->indata->actpointer = 0;
        loc->indata->size = stepsize;

        /* open the file */
        loc->state->fp = fopen_with_path(PARAM(file), "r");
        loc->state->atend = 0;
        if (!loc->state->fp) {
            char *lbuffer;
            lbuffer = getenv("NGSPICE_INPUT_DIR");
            if (lbuffer && *lbuffer) {
                char *p;
                if ((p = (char *) malloc(strlen(lbuffer) +
                        strlen(DIR_PATHSEP) + strlen(PARAM(file)) + 1)) ==
                        (char *) NULL) {
                    cm_message_send("Unable to allocate buffer "
                            "for building file name in cm_filesource()");
                }
                else {
                    sprintf(p, "%s%s%s", lbuffer, DIR_PATHSEP, PARAM(file));
                    loc->state->fp = fopen(p, "r");
                    free(p);
                }
            }
            if (!loc->state->fp) {
                cm_message_printf("cannot open file %s", PARAM(file));
                loc->state->atend = 1;
            }
        }
        /* read, preprocess and store the data */
        amplscalesize = PARAM_NULL(amplscale) ? 0 : PARAM_SIZE(amplscale);
        amploffssize = PARAM_NULL(amploffset) ? 0 : PARAM_SIZE(amploffset);
        count = 0;
        while (!loc->state->atend) {
            char line[512];
            char *cp, *cpdel;
            char *cp2;
            double t, tprev = 0;
            int i;
            if (!fgets(line, sizeof(line), loc->state->fp)) {
                loc->state->atend = 1;
                break;
            }
            if ((cpdel = cp = strdup(line)) == (char *) NULL) {
                cm_message_send("Unable to duplicate string "
                        "cm_filesource()");
                loc->state->atend = 1;
                break;
            }

            /* read the time channel; update the time difference */
            while (*cp && isspace_c(*cp))
                ++cp;
            if (*cp == '*' || *cp == '#' || *cp == ';') {
                free(cpdel);
                continue;
            }
            t = strtod(cp, &cp2);
            if (cp2 == cp) {
                free(cpdel);
                continue;
            }
            cp = cp2;
            if (!PARAM_NULL(timescale))
                t *= PARAM(timescale);
            if (!PARAM_NULL(timerelative) && PARAM(timerelative) == MIF_TRUE)
                t += tprev;
            else if (!PARAM_NULL(timeoffset))
                t += PARAM(timeoffset);

            tprev = t;

            /* before storing, check if vector size is large enough.
               If not, add another 1000*size doubles */
            if (count > (int) loc->indata->vecallocated - size) {
                loc->indata->vecallocated += (size_t) (size * 1000);
                void * const p = realloc(loc->indata->datavec,
                        sizeof(double) * loc->indata->vecallocated);
                if (p == NULL) {
                    cm_message_printf("cannot allocate enough memory");
                    break; // loc->state->atend = 1;
                }
                loc->indata->datavec = (double *) p;
            }
            loc->indata->datavec[count++] = t;

            /* read the data channels; update the amplitude difference of each channel */
            for (i = 0; i < size; ++i) {
                while (*cp && (isspace_c(*cp) || *cp == ','))
                    ++cp;
                t = strtod(cp, &cp2);
                if (cp2 == cp)
                    break;
                cp = cp2;
                if (i < amplscalesize)
                    t *= PARAM(amplscale[i]);
                if (i < amploffssize)
                    t += PARAM(amploffset[i]);
                loc->indata->datavec[count++] = t;
            }
            free(cpdel);
        }
        loc->indata->maxoccupied = count;

        if(loc->state->fp) {
            fclose(loc->state->fp);
            loc->state->fp = NULL;
        }
        /* set the start time data */
        loc->timeinterval[0] = loc->indata->datavec[loc->indata->actpointer];
        loc->timeinterval[1] = loc->indata->datavec[loc->indata->actpointer + stepsize];
    }

    loc = STATIC_VAR (locdata);

    /* The file pointer is at the same position it was for the last simulator TIME ...
     * If TIME steps backward, for example due to a second invocation of a 'tran' analysis
     *   step back in datavec[loc->indata->actpointer] .
     */
    if (TIME < loc->timeinterval[0]) {
        while (TIME < loc->indata->datavec[loc->indata->actpointer] && loc->indata->actpointer >= 0)
            loc->indata->actpointer -= stepsize;
        loc->timeinterval[0] = loc->indata->datavec[loc->indata->actpointer];
        loc->timeinterval[1] = loc->indata->datavec[loc->indata->actpointer + stepsize];
    }

    while (TIME > loc->timeinterval[1]) {
        loc->indata->actpointer += stepsize;
        if (loc->indata->actpointer > loc->indata->maxoccupied) {
            /* we are done */
            return;
        }
        loc->timeinterval[1] = loc->indata->datavec[loc->indata->actpointer + stepsize];
        loc->timeinterval[0] = loc->indata->datavec[loc->indata->actpointer];
    }

    for (j = 0; j < size; j++) {
        loc->amplinterval[2 * j] = loc->indata->datavec[loc->indata->actpointer + j + 1];
        loc->amplinterval[2 * j + 1] = loc->indata->datavec[loc->indata->actpointer + stepsize + j + 1];
    }

    if (loc->timeinterval[0] <= TIME && TIME <= loc->timeinterval[1]) {
        if (!PARAM_NULL(amplstep) && PARAM(amplstep) == MIF_TRUE) {
            int i;
            for (i = 0; i < size; ++i)
                OUTPUT(out[i]) = loc->amplinterval[2 * i];
        } else {
            double mul0 = (loc->timeinterval[1] - TIME) / (loc->timeinterval[1] - loc->timeinterval[0]);
            double mul1 = 1.0 - mul0;
            int i;
            for (i = 0; i < size; ++i)
                OUTPUT(out[i]) = mul0 * loc->amplinterval[2 * i] + mul1 * loc->amplinterval[2 * i + 1];
        }
    } else {
        int i;
        for (i = 0; i < size; ++i)
            OUTPUT(out[i]) = loc->amplinterval[2 * i + 1];
    }
} /* end of function cm_filesource */



static void cm_filesource_callback(ARGS, Mif_Callback_Reason_t reason)
{
    switch (reason) {
        case MIF_CB_DESTROY: {
            Local_Data_t *loc = (Local_Data_t *) STATIC_VAR(locdata);
            if (loc == (Local_Data_t *) NULL) {
                break;
            }

            if (loc->state != (struct filesource_state *) NULL) {
                if (loc->state->fp != (FILE *) NULL) {
                    fclose(loc->state->fp);
                }
                free(loc->state);
            }

            if (loc->amplinterval != (double *) NULL) {
                free(loc->amplinterval);
            }

            if (loc->timeinterval != (double *) NULL) {
                free(loc->timeinterval);
            }

            if (loc->indata) {
                if (loc->indata->datavec) {
                    free(loc->indata->datavec);
                }
                free(loc->indata);
            }

            free(loc);

            STATIC_VAR(locdata) = NULL;
            break;
        }
    }
} /* end of function cm_filesource_callback */




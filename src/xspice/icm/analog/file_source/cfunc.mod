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
    11 Aug 2025     Holger Vogt
    01 Aug 2026     Holger Vogt


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

#define OK 0
#define FAIL 1

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

typedef char line_t[82]; /* A SPICE size line. <= 80 characters plus '\n\0' */

           
/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/


/*=== Static CNV_get_spice_value ROUTINE =============*/

/*
   Function takes as input a string token from a SPICE
deck and returns a floating point equivalent value.
*/


static int cnv_get_spice_value(
char    *str,        /* IN - The value text e.g. 1.2K */
double  *p_value )   /* OUT - The numerical value     */
{


    /* the following were "int4" devices - jpm */
    size_t len;
    size_t i;
    int    n_matched;

    line_t  val_str;

    /*char    *suffix;*/
    char    c = ' ';
    char    c1;

    double  scale_factor;
    double  value;


    /* Scan the input string looking for an alpha character that is not  */
    /* 'e' or 'E'.  Such a character is assumed to be an engineering     */
    /* suffix as defined in the Spice 2G.6 user's manual.                */

    len = strlen(str);
    if( len > (sizeof(val_str) - 1))
        len = sizeof(val_str) - 1;

    for(i = 0; i < len; i++) {
        c = str[i];
        if( isalpha_c(c) && (c != 'E') && (c != 'e') )
            break;
        else if( isspace_c(c) )
            break;
        else
            val_str[i] = c;
    }
    val_str[i] = '\0';


    /* Determine the scale factor */

    if( (i >= len) || (! isalpha_c(c)) )
        scale_factor = 1.0;
    else {

        if(isupper_c(c))
            c = tolower_c(c);

        switch(c) {

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

        case 'a':
            scale_factor = 1.0e-18;
            break;

        case 'm':
            i++;
            if(i >= len) {
                scale_factor = 1.0e-3;
                break;
            }
            c1 = str[i];
            if(! isalpha_c(c1)) {
                scale_factor = 1.0e-3;
                break;
            }
            if(islower_c(c1))
                c1 = toupper_c(c1);
            if(c1 == 'E')
                scale_factor = 1.0e6;
            else if(c1 == 'I')
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

    n_matched = sscanf(val_str,"%le",&value);

    if(n_matched < 1) {
        *p_value = 0.0;
        return(FAIL);
    }

    *p_value = value * scale_factor;
    return(OK);
}



/*=== Static CNVgettok ROUTINE ================*/
/*
Get the next token from the input string.  The input string pointer
is advanced to the following token and the token from the input
string is copied to malloced storage and a pointer to that storage
is returned.  The original input string is undisturbed.
*/

#include <stdlib.h>

static char  *CNVgettok(char **s)

{

    char    *buf;       /* temporary storage to copy token into */
    /*char    *temp;*/      /* temporary storage to copy token into */
    char    *ret_str;   /* storage for returned string */

    int     i;

    /* allocate space big enough for the whole string */

    buf = (char *) malloc(strlen(*s) + 1);

    /* skip over any white space */

    while(isspace_c(**s) || (**s == '=') ||
          (**s == '(') || (**s == ')') || (**s == ','))
          (*s)++;

    /* isolate the next token */

    switch(**s) {

    case '\0':           /* End of string found */
        if(buf) free(buf);
        return(NULL);


    default:             /* Otherwise, we are dealing with a    */
                         /* string representation of a number   */
                         /* or a mess o' characters.            */
        i = 0;
        while( (**s != '\0') &&
               (! ( isspace_c(**s) || (**s == '=') ||
                    (**s == '(') || (**s == ')') ||
                    (**s == ',')
             ) )  ) {
            buf[i] = **s;
            i++;
            (*s)++;
        }
        buf[i] = '\0';
        break;
    }

    /* skip over white space up to next token */

    while(isspace_c(**s) || (**s == '=') ||
          (**s == '(') || (**s == ')') || (**s == ','))
          (*s)++;

    /* make a copy using only the space needed by the string length */


    ret_str = (char *) malloc(strlen(buf) + 1);
    ret_str = strcpy(ret_str,buf);

    if(buf) free(buf);

    return(ret_str);
}


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
        double tprev; /* store the previous time, used when time relative is set */
        bool terr = MIF_FALSE; /* Cumulative warning message upon error during time reading */
        bool derr = MIF_FALSE; /* Cumulative warning message upon error during data reading */

        /* add time offset only once to tprev */
        if (!PARAM_NULL(timeoffset)) {
            tprev = PARAM(timeoffset);
        }
        else
            tprev = 0.;


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
                cm_cexit(1);
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
            double t = 0, d = 0;
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
            if (*cp == '*' || *cp == '#' || *cp == ';' || *cp == '\0') {
                free(cpdel);
                continue;
            }
            char *ncp = CNVgettok(&cp);
            int ret = cnv_get_spice_value(ncp, &t);
            free(ncp);
            if (ret == FAIL) {
                free(cpdel);
                terr = MIF_TRUE;
                continue;
            }
            if (!PARAM_NULL(timescale))
                t *= PARAM(timescale);
            if (!PARAM_NULL(timerelative) && PARAM(timerelative) == MIF_TRUE)
                t += tprev;
            else if (!PARAM_NULL(timeoffset))
                t += PARAM(timeoffset);

            tprev = t;

            /* before storing, check if vector size is large enough.
               If not, add another 1000*size doubles.  Each record appended below
               is a full stepsize (= size + 1: one timepoint plus `size` channel
               values), so reserve stepsize -- reserving only `size` left room for
               one fewer value than is written and overran the buffer by one
               double at the allocation boundary. */
            if (count > (int) loc->indata->vecallocated - stepsize) {
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
                char *ncp = CNVgettok(&cp);
                int ret = cnv_get_spice_value(ncp, &d);
                free(ncp);
                if (ret == FAIL) {
                    derr = MIF_TRUE;
                    break;
                }
                if (i < amplscalesize)
                    d *= PARAM(amplscale[i]);
                if (i < amploffssize)
                    d += PARAM(amploffset[i]);
                loc->indata->datavec[count++] = d;
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

        if (terr)
            cm_message_printf("WARNING: some error occurred during reading the time values");
        if (derr)
            cm_message_printf("WARNING: some error occurred during reading the data values");
    }

    loc = STATIC_VAR (locdata);

    /* The file pointer is at the same position it was for the last simulator TIME ...
     * If TIME steps backward, for example due to a second invocation of a 'tran' analysis
     *   step back in datavec[loc->indata->actpointer] .
     */
    if (TIME < loc->timeinterval[0] && loc->indata->actpointer >= stepsize) {
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




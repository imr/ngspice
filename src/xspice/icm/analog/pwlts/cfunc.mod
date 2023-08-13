/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE pwlts/cfunc.mod

Public Domain


AUTHORS

    Original author of pwl
    19 Apr 1991     Jeffrey P. Murray

Pwl with time input and smoothing: pwlts
     9 Sep 2022    Holger Vogt

SUMMARY

    This file contains the model-specific routines used to
    functionally describe the pwlts (piece-wise linear time based) code model.


INTERFACES

    FILE                 ROUTINE CALLED

    CMutil.c             void cm_smooth_corner();

    CMmacros.h           cm_message_send();


REFERENCED FILES

    Inputs from and outputs to ARGS structure.


NON-STANDARD FEATURES

    NONE

===============================================================================*/

/*=== INCLUDE FILES ====================*/

#include <math.h>



/*=== CONSTANTS ========================*/

#define FRACTION 0.30
#define EPSILON 1.0e-9



/*=== MACROS ===========================*/




/*=== LOCAL VARIABLES & TYPEDEFS =======*/




/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/



#include <stdlib.h>


/*==============================================================================

FUNCTION void cm_pwlts

AUTHORS

    9 Sep 2022    Holger Vogt

MODIFICATIONS


SUMMARY

    This function implements the pwlts code model.

INTERFACES

    FILE                 ROUTINE CALLED

    CMutil.c             void cm_smooth_corner();

    CMmacros.h           cm_message_send();


RETURNED VALUE

    Returns outputs via ARGS structure.

GLOBAL VARIABLES

    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

static void
cm_pwlts_callback(ARGS, Mif_Callback_Reason_t reason)
{
    switch (reason) {
        case MIF_CB_DESTROY: {
            double *last_x_value = STATIC_VAR (last_x_value);
            double *x = STATIC_VAR (x);
            double *y = STATIC_VAR (y);
            if (last_x_value)
                free(last_x_value);
            if (x)
                free(x);
            if (y)
                free(y);
            STATIC_VAR (last_x_value) = NULL;
            STATIC_VAR (x) = NULL;
            STATIC_VAR (y) = NULL;
            break;
        }
    }
}

/*=== CM_PWLTS ROUTINE ================*/

void cm_pwlts(ARGS)  /* structure holding parms,
                                   inputs, outputs, etc.     */
{
    int i;                  /* generic loop counter index */
    int size;               /* size of the x_array        */

    double input_domain;    /* smoothing range */
    double *x;              /* pointer to the x-coordinate array */
    double *y;              /* pointer to the y-coordinate array */
    double lower_seg;       /* x segment below which input resides */
    double upper_seg;       /* x segment above which the input resides */
    double lower_slope;     /* slope of the lower segment */
    double upper_slope;     /* slope of the upper segment */
    double x_input;         /* input */
    double out = 0.0;       /* output
                             * Init to 0 to suppress compiler warning */
    double dout_din = 0.0;  /* partial derivative of the output wrt input.
                             * Init to 0 to suppress compiler warning */
    double threshold_lower; /* value below which the output begins smoothing */
    double threshold_upper; /* value above which the output begins smoothing */
    double test1;           /* debug testing value */
    double test2;           /* debug testing value */
    double *last_x_value;   /* static variable for limiting */

    CALLBACK = cm_pwlts_callback;

    char *allocation_error="\n***ERROR***\nPWL: Allocation calloc failed!\n";
    char *limit_error="\n***ERROR***\nPWL: Violation of 50% rule in breakpoints!\n";

    /* Retrieve frequently used parameters... */

    input_domain = PARAM(input_domain);

    /* size including space for two additional x,y pairs */
    size = PARAM_SIZE(x_array) + 2;


    /* First pass:
    Allocate storage for previous value.
    Allocate storage for x an y input arrays
    Read input array and store from
    Add additional x,y pair at beginning and end of x, y arrays:
    */
    if (INIT==1) {
        /* Allocate storage for last_x_value */
        STATIC_VAR(last_x_value) = (double *) malloc(sizeof(double));
        last_x_value = (double *) STATIC_VAR(last_x_value);

        /* Allocate storage for breakpoint domain & range values */
        STATIC_VAR(x) = (double *) calloc((size_t) size, sizeof(double));
        x = (double *) STATIC_VAR(x);
        if (!x) {
            cm_message_send(allocation_error);
        }

        STATIC_VAR(y) = (double *) calloc((size_t) size, sizeof(double));
        y = (double *) STATIC_VAR(y);
        if (!y) {
            cm_message_send(allocation_error);
        }

        /* Retrieve x and y values. */
        for (i=1; i<size-1; i++) {
            x[i] = PARAM(x_array[i - 1]);
            y[i] = PARAM(y_array[i - 1]);
        }
        /* Add additional leading and trailing values */
        x[0] = 2. * x[1] - x[2];
        x[size - 1] = 2. * x[size - 2] - x[size - 3];
        if (PARAM(limit) == MIF_TRUE) {
            /* const additional y values */
            y[0] = y[1];
            y[size - 1] = y[size - 2];
        }
        else {
            /* linearily extrapolated additional y values */
            y[0] = 2. * y[1] - y[2];
            y[size - 1] = 2. * y[size - 2] - y[size - 3];
        }
        /* debug printout
        for (i = 0; i < size; i++)
           fprintf(stderr, "%e ", y[i]);
        fprintf(stderr, "\n");
        for (i = 0; i < size; i++)
           fprintf(stderr, "%e ", x[i]);
        fprintf(stderr, "\n"); */
    }
    else {

        last_x_value = (double *) STATIC_VAR(last_x_value);

        x = (double *) STATIC_VAR(x);

        y = (double *) STATIC_VAR(y);

    }

    /* See if input_domain is absolute...if so, test against   */
    /* breakpoint segments for violation of 50% rule...        */
    if (PARAM(fraction) == MIF_FALSE) {
        if ( 3 < size ) {
            for (i=1; i<(size-2); i++) {
                /* Test for overlap...0.999999999 factor is to      */
                /* prevent floating point problems with comparison. */
                if ( (test1 = x[i+1] - x[i]) <
                     (test2 = 0.999999999 * (2.0 * input_domain)) ) {
                    cm_message_send(limit_error);
                }
            }
        }

    }

    /* Retrieve x_input value as current simulation time. */
    x_input = TIME;


    /* If this is the first call, set *last_x_value to x_input */
    if (INIT == 1)
        *last_x_value=x_input;


    /*** Add debugging printf statement ***/
    /* printf("Last x_input=%e, Current x_input=%e,\n",
            *last_x_value,x_input);
    */

    /* Determine segment boundaries within which x_input resides */

    if (x_input <= (x[0] + x[1])/2.0) {/*** x_input below lowest midpoint ***/
        dout_din = (y[1] - y[0])/(x[1] - x[0]);


        /* Compute new output */
        out = y[0] + (x_input - x[0]) * dout_din;
    }
    else {
        if (x_input >= (x[size-2] + x[size-1])/2.0) {
                                   /*** x_input above highest midpoint ***/
            dout_din = (y[size-1] - y[size-2]) /
                          (x[size-1] - x[size-2]);

            out = y[size-1] + (x_input - x[size-1]) * dout_din;
        }
        else { /*** x_input within bounds of end midpoints...     ***/
               /*** must determine position progressively & then  ***/
               /*** calculate required output.                    ***/

            for (i = 1; i < size - 1; i++) {

                if (x_input < (x[i] + x[i+1])/2.0) {
                                   /* approximate position known...          */

                    lower_seg = (x[i] - x[i-1]);
                    upper_seg = (x[i+1] - x[i]);


                    /* Calculate input_domain about this region's breakpoint.*/

                    if (PARAM(fraction) == MIF_TRUE) {  /* Translate input_domain */
                                                  /* into an absolute....   */
                        if ( lower_seg <= upper_seg )          /* Use lower  */
                                                               /* segment    */
                                                               /* for % calc.*/
                            input_domain = input_domain * lower_seg;
                        else                                   /* Use upper  */
                                                               /* segment    */
                                                               /* for % calc.*/
                            input_domain = input_domain * upper_seg;
                    }

                    /* Set up threshold values about breakpoint... */
                    threshold_lower = x[i] - input_domain;
                    threshold_upper = x[i] + input_domain;

                    /* Determine where x_input is within region & determine */
                    /* output and partial values....                        */
                    if (x_input < threshold_lower) { /* Lower linear region */
                        dout_din = (y[i] - y[i-1])/lower_seg;

                        out = y[i] + (x_input - x[i]) * dout_din;
                    }
                    else {
                        if (x_input < threshold_upper) { /* Parabolic region */
                            lower_slope = (y[i] - y[i-1])/lower_seg;
                            upper_slope = (y[i+1] - y[i])/upper_seg;

                            cm_smooth_corner(x_input,x[i],y[i],input_domain,
                                        lower_slope,upper_slope,&out,&dout_din);
                        }
                        else {        /* Upper linear region */
                            dout_din = (y[i+1] - y[i])/upper_seg;
                            out = y[i] + (x_input - x[i]) * dout_din;
                        }
                    }
                    break;  /* Break search loop...x_input has been found, */
                            /* and out and dout_din have been assigned.    */
                }
            }
        }
    }
    /* returns time 0 value for dc and 0 for ac simulation */
    OUTPUT(out) = out;
}


/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE pswitch/cfunc.mod

3-Clause BSD

Copyright 2020 The ngspice team


AUTHORS

    27 September 2020     Holger Vogt


MODIFICATIONS

    03 June 2021    Yurii Demchyna


SUMMARY

    This file contains the functional description of the pswitch
    code model.


INTERFACES

    FILE                 ROUTINE CALLED

    CMmacros.h           cm_message_send();


REFERENCED FILES

    Inputs from and outputs to ARGS structure.


NON-STANDARD FEATURES

    NONE

===============================================================================*/

/*=== INCLUDE FILES ====================*/

#include <stdlib.h>
#include <math.h>



/*=== CONSTANTS ========================*/




/*=== MACROS ===========================*/




/*=== LOCAL VARIABLES & TYPEDEFS =======*/

typedef struct {

    double logmean;      /* log-mean of resistor values */
    double logratio;     /* log-ratio of resistor values */
    double cntl_mean;    /* mean of control values */
    double cntl_diff;    /* diff of control values */
    double intermediate; /* intermediate value used to calculate
                the resistance of the switch when the
                controlling voltage is between cntl_on
                and cntl_of */
    double cntl_on;      /* voltage above which switch come on */
    double cntl_off;     /* voltage below the switch has resistance roff */
    double c1;           /* some constants */
    double c2;
    double c3;
} Local_Data_t;


/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/

static void
cm_pswitch_callback(ARGS, Mif_Callback_Reason_t reason)
{
    switch (reason) {
        case MIF_CB_DESTROY: {
            Local_Data_t *loc = STATIC_VAR (locdata);
	    if (loc) {
                free(loc);
                STATIC_VAR (locdata) = NULL;
            }
            break;
        }
    }
}






/*==============================================================================

FUNCTION cm_pswitch()

AUTHORS

    27 September 2020     Holger Vogt

MODIFICATIONS

SUMMARY

    This function implements the pswitch code model.

INTERFACES

    FILE                 ROUTINE CALLED

    CMmacros.h           cm_message_send();

RETURNED VALUE

    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES

    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== CM_PSWITCH ROUTINE ===*/



void cm_pswitch(ARGS)  /* structure holding parms,
                                          inputs, outputs, etc.     */
{
    double cntl_on;      /* voltage above which switch come on */
    double cntl_off;     /* voltage below the switch has resistance roff */
    double r_on;         /* on resistance */
    double r_off;        /* off resistance */
    double r_cntl_in;    /* input resistance for control terminal */
    double logmean;      /* log-mean of resistor values */
    double logratio;     /* log-ratio of resistor values */
    double cntl_mean;    /* mean of control values */
    double cntl_diff;    /* diff of control values */
    double intermediate; /* intermediate value used to calculate
                            the resistance of the switch when the
                            controlling voltage is between cntl_on
                            and cntl_of */
    double r;            /* value of the resistance of the switch */
    double pi_pvout;     /* partial of the output wrt input       */
    double pi_pcntl;     /* partial of the output wrt control input */

    Mif_Complex_t ac_gain;



    Local_Data_t *loc;    /* Pointer to local static data, not to be included
                                       in the state vector */


    /* Retrieve frequently used parameters... */

    r_on = PARAM(r_on);
    r_off = PARAM(r_off);
    r_cntl_in = PARAM(r_cntl_in);

    r_on = (r_on < 1.0e-3) ? 1.0e-3 : r_on;  /* Set minimum 'ON' resistance */
    r_off = (r_off > 1.0e12) ? 1.0e12 : r_off;  /* Set maximum 'OFF' resistance */


    if(INIT == 1) { /* first time through, allocate memory, set static parameters */
        char *cntl_error = "\n*****ERROR*****\nPSWITCH: CONTROL voltage delta less than 1.0e-12\n";

        cntl_on = PARAM(cntl_on);
        cntl_off = PARAM(cntl_off);
        if( (fabs(cntl_on - cntl_off) < 1.0e-12) ) {
            cntl_on += 0.001;
            cntl_off -= 0.001;
        }

        CALLBACK = cm_pswitch_callback;

        /*** allocate static storage for *loc ***/
        STATIC_VAR (locdata) = calloc (1 , sizeof ( Local_Data_t ));
        loc = STATIC_VAR (locdata);

        loc->cntl_on = cntl_on;
        loc->cntl_off = cntl_off;

        if ( PARAM(log) == MIF_TRUE ) {   /* Logarithmic Variation in 'R' */
            if (cntl_on > cntl_off)
            {
                cntl_on = 1;
                cntl_off = 0;
            }
            else
            {
                cntl_on = 0;
                cntl_off = 1;
            }

            loc->logmean = log(sqrt(r_on * r_off));
            loc->logratio = log(r_on / r_off);
            loc->cntl_mean = 0.5;
            loc->cntl_diff = cntl_on - cntl_off;
            loc->intermediate = loc->logratio / loc->cntl_diff;
            loc->c1 = 1.5 * loc->logratio / loc->cntl_diff;
            loc->c3 = 2. * loc->logratio / (loc->cntl_diff * loc->cntl_diff * loc->cntl_diff); //pow(loc->cntl_diff, 3);
            loc->c2 = 3 * loc->c3;
        } else {
            loc->cntl_diff = cntl_on - cntl_off;
            loc->intermediate = (r_on - r_off) / (cntl_on - cntl_off);
        }
    }

    loc = STATIC_VAR (locdata);

    cntl_on = loc->cntl_on;
    cntl_off = loc->cntl_off;
    if ( PARAM(log) == MIF_TRUE ) {   /* Logarithmic Variation in 'R' */
        logmean = loc->logmean;
        logratio = loc->logratio;
        cntl_mean = loc->cntl_mean;
        cntl_diff = loc->cntl_diff;
        intermediate = loc->intermediate;
        double inmean;// = INPUT(cntl_in) - cntl_mean;
        int outOfLimit = 0;
        if (cntl_on > cntl_off) {
            inmean = (INPUT(cntl_in) - cntl_off) / (cntl_on - cntl_off) - cntl_mean;
            if (INPUT(cntl_in) > cntl_on) {
                r = r_on;
                outOfLimit = 1;
            }
            else if (INPUT(cntl_in) < cntl_off) {
                r = r_off;
                outOfLimit = 1;
            }
            else {
                r = exp(logmean + loc->c1 * inmean - loc->c3 * inmean * inmean * inmean);
                if(r<r_on) r=r_on;/* minimum resistance limiter */
            }
        } else {
            inmean = (cntl_on - INPUT(cntl_in)) / (cntl_on - cntl_off) - cntl_mean;
            if (INPUT(cntl_in) < cntl_on) {
                r = r_on;
                outOfLimit = 1;
            }
            else if (INPUT(cntl_in) > cntl_off) {
                r = r_off;
                outOfLimit = 1;
            }
            else {
                r = exp(logmean + loc->c1 * inmean - loc->c3 * inmean * inmean * inmean);
                if(r<r_on) r=r_on;/* minimum resistance limiter */
            }
        }

        pi_pcntl = INPUT(out) / r * (loc->c2 * inmean * inmean - loc->c1);
        if(1 == outOfLimit){
            pi_pcntl = 0;
        }
        pi_pvout = 1.0 / r;

    }
    else {                      /* Linear Variation in 'R' */
        intermediate = loc->intermediate;
        cntl_diff = loc->cntl_diff;
        if (cntl_diff >=0) {
            if (INPUT(cntl_in) < cntl_off) {
                r = r_off;
                pi_pcntl = 0;
            }
            else if (INPUT(cntl_in) > cntl_on) {
                r = r_on;
                pi_pcntl = 0;
            }
            else {
                r = INPUT(cntl_in) * intermediate + ((r_off*cntl_on -
                    r_on*cntl_off) / cntl_diff);
                pi_pcntl = -intermediate * INPUT(out) / (r*r);
            }
        }
        else {
            if (INPUT(cntl_in) > cntl_off) {
                r = r_off;
                pi_pcntl = 0;
            }
            else if (INPUT(cntl_in) < cntl_on) {
                r = r_on;
                pi_pcntl = 0;
            }
            else {
                r = INPUT(cntl_in) * intermediate + ((r_off*cntl_on -
                    r_on*cntl_off) / cntl_diff);
                pi_pcntl = -intermediate * INPUT(out) / (r*r);
            }        
        }
        if(r<=1.0e-9) r=1.0e-9;/* minimum resistance limiter */
        pi_pvout = 1.0 / r;
    }

    if(ANALYSIS != MIF_AC) {            /* Output DC & Transient Values  */
        OUTPUT(out) = INPUT(out) / r;
        OUTPUT(cntl_in) = INPUT(cntl_in) / r_cntl_in;
//        PARTIAL(out,out) = pi_pvout;
//        PARTIAL(out,cntl_in) = pi_pcntl;
//        PARTIAL(cntl_in,cntl_in) = 1 / r_cntl_in;
//        PARTIAL(cntl_in,out) = 0;           /* cntl input resistance is
//                                              independent to out port */
        cm_analog_auto_partial();

    /* Note that the minus signs are required  because current is positive
       flowing INTO rather than OUT OF a component node.       */
    }
    else {                              /*   Output AC Gain Values      */
        ac_gain.real = -pi_pvout;           /* See comment on minus signs above...  */
        ac_gain.imag= 0.0;
        AC_GAIN(out,out) = ac_gain;

        ac_gain.real = -pi_pcntl;
        ac_gain.imag= 0.0;
        AC_GAIN(out,cntl_in) = ac_gain;
    }
}


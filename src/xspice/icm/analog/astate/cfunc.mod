/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE astate/cfunc.mod

3-clause BSD

Copyright 2025
The ngspice team

AUTHORS

    20 September 2025     Holger Vogt


MODIFICATIONS



SUMMARY

    This file contains the functional description of the analog
    state code model.
    It takes an input node, stores its value (voltage or current)
    during the last three time steps and returns a value at the output,
    delayed by 0 to 3 steps, depending on the model parameter astate_no.


INTERFACES

    FILE                 ROUTINE CALLED



REFERENCED FILES

    Inputs from and outputs to ARGS structure.


NON-STANDARD FEATURES

    NONE

===============================================================================*/

/*=== INCLUDE FILES ====================*/

#include <stdlib.h>


/*=== CONSTANTS ========================*/




/*=== MACROS ===========================*/




/*=== LOCAL VARIABLES & TYPEDEFS =======*/

typedef struct {
    double    state1;       /* first state value */
    double    state2;       /* second state value */
    double    state3;       /* third state value */
    double    outval;       /* output value */
    double    xval1;        /* first x value */
    double    xval2;        /* second x value */
    double    xval3;        /* third x value */
} stLocal_Data_t;





/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/

static void cm_astate_callback(ARGS, Mif_Callback_Reason_t reason);

/*=== CM_STATE ROUTINE ===*/

void cm_astate(ARGS)
{
    int state_number = 0;
    double outval = 0.0;

    stLocal_Data_t *loc;        /* Pointer to local static data, not to be included
                                       in the state vector */


    if (ANALYSIS != MIF_AC) {     /**** only Transient Analysis and dc ****/

        /** INIT: allocate storage **/

        if (INIT==1) {

            CALLBACK = cm_astate_callback;

            /*** allocate static storage for *loc ***/
            if ((loc = (stLocal_Data_t *) (STATIC_VAR(locdata) = calloc(1,
                    sizeof(stLocal_Data_t)))) == (stLocal_Data_t *) NULL) {
                cm_message_send("Unable to allocate Local_Data_t "
                    "in cm_astate()");
                return;
            }
            loc->state1 = 0;
            loc->state2 = 0;
            loc->state3 = 0;
            loc->outval = 0;
            loc->xval1 = 0;
            loc->xval2 = 0;
            loc->xval3 = 0;
        }

        /* retrieve previous values */

        loc = STATIC_VAR (locdata);

        state_number = PARAM(astate_no);

        if (state_number == 0) {
            OUTPUT(out) = INPUT(in);
            return;
        }

        if (TIME == loc->xval1) {
            switch(state_number) {
                case 1:
                    loc->outval = loc->state1;
                    break;
                case 2:
                    loc->outval = loc->state2;
                    break;
                case 3:
                    loc->outval = loc->state3;
                    break;
                default: 
                    loc->outval = INPUT(in);
                    break;
            }
        }
        else if (TIME > loc->xval1) {
            loc->state3 = loc->state2;
            loc->state2 = loc->state1;
            loc->state1 = INPUT(in);

            loc->xval3 = loc->xval2;
            loc->xval2 = loc->xval1;
            loc->xval1 = TIME;
        }
        /* initial time iteration */
        else if (TIME == 0.0) {
            loc->state1 = loc->outval = INPUT(in);
            loc->xval1 = 0.0;
        }
        /* time step rejected ? */
        else if (TIME < loc->xval1){
            loc->state1 = INPUT(in);
            loc->xval1 = TIME;
        }
        /* output */
        if (ANALYSIS == MIF_TRAN) {
            OUTPUT(out) = loc->outval;
        }
        else { /* dc */
            OUTPUT(out) = INPUT(in);
        }
    }
    else {
        OUTPUT(out) = INPUT(in);
    }
}

/* free the memory created locally */
static void cm_astate_callback(ARGS, Mif_Callback_Reason_t reason)
{
    switch (reason) {
        case MIF_CB_DESTROY: {
            stLocal_Data_t *loc = (stLocal_Data_t *) STATIC_VAR(locdata);
            if (loc == (stLocal_Data_t *) NULL) {
                break;
            }

            free(loc);

            STATIC_VAR(locdata) = NULL;
            break;
        }
    }
} /* end of function cm_astate_callback */

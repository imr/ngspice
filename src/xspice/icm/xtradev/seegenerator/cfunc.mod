/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE seegenerator/cfunc.mod

Public Domain

Universty Duisburg-Essen
Duisburg, Germany
Project Flowspace

AUTHORS                      

    19 May 2025 Holger Vogt


MODIFICATIONS   



SUMMARY

    This file contains the model-specific routines used to
    functionally describe the see (single event effects) generator code model.


INTERFACES       

    FILE                 ROUTINE CALLED     

    CMutil.c             void cm_smooth_corner(); 
                         void cm_smooth_discontinuity();
                         void cm_climit_fcn()

REFERENCED FILES

    Inputs from and outputs to ARGS structure.
                     

NON-STANDARD FEATURES

    NONE

===============================================================================*/

/*=== INCLUDE FILES ====================*/


                                      

/*=== CONSTANTS ========================*/




/*=== MACROS ===========================*/



  
/*=== LOCAL VARIABLES & TYPEDEFS =======*/                         

static void
cm_seegen_callback(ARGS, Mif_Callback_Reason_t reason)
{
    switch (reason) {
        case MIF_CB_DESTROY: {
            double *last_t_value = STATIC_VAR (last_t_value);
            if (last_t_value)
                free(last_t_value);
            STATIC_VAR (last_t_value) = NULL;
            int *pulse_number = STATIC_VAR (pulse_number);
            if (pulse_number)
                free(pulse_number);
            STATIC_VAR (pulse_number) = NULL;
            break;
        }
    }
}
    
           
/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/




                   
/*==============================================================================

FUNCTION void cm_seegen()

AUTHORS                      

    19 May 2025 Holger Vogt

SUMMARY

    This function implements the see generator code model.

INTERFACES       

    FILE                 ROUTINE CALLED     

    CMutil.c             void cm_smooth_corner(); 
                         void cm_smooth_discontinuity();
                         void cm_climit_fcn()

RETURNED VALUE
    
    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== CM_SEEGEN ROUTINE ===*/

void cm_seegen(ARGS)  /* structure holding parms, 
                                       inputs, outputs, etc.     */
{
    double talpha;           /* parameter alpha */
    double tbeta;            /* parameter beta */
    double tdelay;           /* delay until first pulse */
    double inull;            /* max. current of pulse */
    double tperiod;          /* pulse repetition period */
    double out;              /* output current */
    double *last_t_value;    /* static storage of next pulse time */
    int *pulse_number;       /* static storage of next pulse time */
    double tcurr = TIME;     /* current simulation time */


    /* Retrieve frequently used parameters... */

    talpha = PARAM(talpha);
    tbeta = PARAM(tbeta);
    tdelay = PARAM(tdelay);
    tperiod = PARAM(tperiod);
    inull = PARAM(inull);

    if (INIT==1) {
        /* Allocate storage for last_t_value */
        STATIC_VAR(last_t_value) = (double *) malloc(sizeof(double));
        last_t_value = (double *) STATIC_VAR(last_t_value);
        *last_t_value = tdelay;
        STATIC_VAR(pulse_number) = (int *) malloc(sizeof(int));
        pulse_number = (int *) STATIC_VAR(pulse_number);
        *pulse_number = 1;
    }
    else {

        last_t_value = (double *) STATIC_VAR(last_t_value);
        pulse_number = (int *) STATIC_VAR(pulse_number);

        if (tcurr < *last_t_value)
            out = 0;
        else
            out = inull * (exp(-(tcurr-*last_t_value)/talpha) - exp(-(tcurr-*last_t_value)/tbeta));

        if (tcurr > *last_t_value + tperiod * 0.9) {
            *last_t_value = *last_t_value + tperiod;
            (*pulse_number)++;
        }
        if (*pulse_number - 1 < PORT_SIZE(out))
           OUTPUT(out[*pulse_number - 1]) = out;
    }
}




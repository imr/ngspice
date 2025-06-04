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
            double *last_ctrl = STATIC_VAR (last_ctrl);
            if (last_ctrl)
                free(last_ctrl);
            STATIC_VAR (last_ctrl) = NULL;
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

    model source: 
    Ygor Quadros de Aguiar, Frédéric Wrobel. Jean-Luc Autran, Rubén García Alía
    Single-Event Effects, from Space to Accelerator Environments
    Springer 2025

==============================================================================*/

/*=== CM_SEEGEN ROUTINE ===*/

void cm_seegen(ARGS)  /* structure holding parms, 
                                       inputs, outputs, etc.     */
{
    double tfall;            /* pulse fall time */
    double trise;            /* pulse rise time */
    double tdelay;           /* delay until first pulse */
    double inull;            /* max. current of pulse */
    double let;              /* linear energy transfer */
    double cdepth;           /* charge collection depth */
    double angle;            /* particle entrance angle */
    double tperiod;          /* pulse repetition period */
    double ctrlthres;        /* control voltage threshold */
    double ctrl;             /* control input */
    double out;              /* output current */
    double *last_t_value;    /* static storage of next pulse time */
    int *pulse_number;       /* static storage of next pulse time */
    double *last_ctrl;       /* static storage of last ctrl value */
    double tcurr = TIME;     /* current simulation time */

    if (ANALYSIS == MIF_AC) {
        return;
    }

    /* Retrieve frequently used parameters... */

    tfall = PARAM(tfall);
    trise = PARAM(trise);
    tdelay = PARAM(tdelay);
    tperiod = PARAM(tperiod);
    inull = PARAM(inull);
    let = PARAM(let);
    cdepth = PARAM(cdepth);
    angle = PARAM(angle);
    ctrlthres = PARAM(ctrlthres);

    if (PORT_NULL(ctrl))
        ctrl = 1;
    else
        ctrl = INPUT(ctrl);

    if (INIT==1) {
        /* Allocate storage for last_t_value */
        STATIC_VAR(last_t_value) = (double *) malloc(sizeof(double));
        last_t_value = (double *) STATIC_VAR(last_t_value);
        *last_t_value = tdelay;
        STATIC_VAR(pulse_number) = (int *) malloc(sizeof(int));
        pulse_number = (int *) STATIC_VAR(pulse_number);
        *pulse_number = 1;
        STATIC_VAR(last_ctrl) = (double *) malloc(sizeof(double));
        last_ctrl = (double *) STATIC_VAR(last_ctrl);
        *last_ctrl = ctrl;
        /* set breakpoints at new pulse start and pulse maximum times */
        double tatmax = *last_t_value + tfall * trise * log(trise/tfall) / (trise - tfall);
        cm_analog_set_perm_bkpt(*last_t_value);
        cm_analog_set_perm_bkpt(tatmax);
    }
    else {

        last_t_value = (double *) STATIC_VAR(last_t_value);
        pulse_number = (int *) STATIC_VAR(pulse_number);
        last_ctrl = (double *) STATIC_VAR(last_ctrl);

        if (*last_ctrl < ctrlthres && ctrl >= ctrlthres) {
            *last_t_value = *last_t_value + tcurr;
            *last_ctrl = ctrl;
        }

        /* the double exponential current pulse function */
        if (tcurr < *last_t_value)
            out = 0;
        else {
            if (inull == 0) {
                double LETeff = let/cos(angle);
                double Qc = 1.035e-14 * LETeff * cdepth;
                inull = Qc / (tfall - trise);
            }
            out = inull * (exp(-(tcurr-*last_t_value)/tfall) - exp(-(tcurr-*last_t_value)/trise));
        }
        if (tcurr > *last_t_value + tperiod * 0.9) {
            /* return some info */
            cm_message_printf("port no.: %d, port name: out, \nnode names: %s, %s, pulse time: %e",
                *pulse_number, cm_get_node_name("out", *pulse_number - 1), 
                cm_get_neg_node_name("out", *pulse_number - 1), *last_t_value);
            /* set the time for the next pulse */
            *last_t_value = *last_t_value + tperiod;
            /* set breakpoints at new pulse start and pulse maximum times */
            double tatmax = *last_t_value + tfall * trise * log(trise/tfall) / (trise - tfall);
            cm_analog_set_perm_bkpt(*last_t_value);
            cm_analog_set_perm_bkpt(tatmax);
            (*pulse_number)++;
            if (*pulse_number > PORT_SIZE(out)) {
                if (PARAM(perlim) == FALSE)
                    *pulse_number = 1;
                else
                    *last_t_value = 1e12; /* stop any output */
            }
        }
        if (*pulse_number - 1 < PORT_SIZE(out))
           OUTPUT(out[*pulse_number - 1]) = out;
    }
}




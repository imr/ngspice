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

typedef struct pulse_info
{
    double iscaled; /* scaled current pulse for this port */
    double start_time; /* pulse start time for this port */
    double next_start_time; /* next pulse start time for this port */
} pulse_info_t;


/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/

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
            double *last_ctrl = STATIC_VAR (last_ctrl);
            if (last_ctrl)
                free(last_ctrl);
            STATIC_VAR (last_ctrl) = NULL;
            pulse_info_t *pulses = STATIC_VAR (pulses);
            if (pulses)
               free(pulses);
            break;
        }
    }
}

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

    int ports;               /* number of output ports */
    pulse_info_t *allpulses; /* info for pulse on each port */
    bool have_scaled = FALSE;/* TRUE if we want to use scaled pulses */

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

    have_scaled = !PARAM_NULL(scaling) && !PARAM_NULL(scdelay);

    ports = PORT_SIZE(out);

    if (PORT_NULL(ctrl))
        ctrl = 1;
    else
        ctrl = INPUT(ctrl);

    if (INIT==1) {

        int i;
        double sum = 0;

        if (have_scaled && PARAM_SIZE(scaling) != ports) {
            cm_message_send("Error: Number of Output ports and Scaling don't match\n");
            cm_cexit(1);
        }

        if (have_scaled && PARAM_SIZE(scdelay) != ports) {
            cm_message_send("Error: Number of Output ports and SCdelay don't match\n");
            cm_cexit(1);
        }

        if (!have_scaled && 5 * (trise + tfall) > tperiod) {
            cm_message_send("\nError: tperiod should be at least 5 times the sum of trise and tfall\n");
            cm_cexit(1);
        }

        CALLBACK = cm_seegen_callback;

        if (have_scaled) {
            int j;
            double del = 1e12;

            cm_message_send("Use the scaling option\n");

            allpulses = STATIC_VAR(pulses) = (pulse_info_t *) malloc(ports * sizeof(pulse_info_t));

            /* parameter inull not specified, calculate it */
            if (inull == 0) {
                double LETeff = let/cos(angle);
                double Qc = 1.035e-14 * LETeff * cdepth;
                inull = Qc / (tfall - trise);
            }

            /* pulse currents are scaled, and find minimum time delay */
            for (i = 0; i < ports; i++){
                sum += PARAM(scaling[i]);
            }
            if (sum == 0.) {
                cm_message_send("Error: Scaling parameters are zero\n");
                cm_cexit(1);
            }

            for (i = 0; i < ports; i++){
                allpulses[i].iscaled = PARAM(scaling[i]) / sum * inull;
                allpulses[i].start_time = tdelay + PARAM(scdelay[i]);
                double tatmax = allpulses[i].start_time + tfall * trise * log(trise/tfall) / (trise - tfall);
                cm_analog_set_perm_bkpt(allpulses[i].start_time);
                cm_analog_set_perm_bkpt(tatmax);
            }
        }
        else {
            /* Allocate storage for last_t_value */
            STATIC_VAR(last_t_value) = (double *) malloc(sizeof(double));
            last_t_value = (double *) STATIC_VAR(last_t_value);
            /* no start if ctrl is set */
            if (PORT_NULL(ctrl))
                *last_t_value = tdelay;
            else
                *last_t_value = 1e12;
            STATIC_VAR(last_ctrl) = (double *) malloc(sizeof(double));
            last_ctrl = (double *) STATIC_VAR(last_ctrl);
            *last_ctrl = ctrl;
            STATIC_VAR(pulse_number) = (int *) malloc(sizeof(int));
            pulse_number = (int *) STATIC_VAR(pulse_number);
            *pulse_number = 1;

            /* set breakpoints at first pulse start and pulse maximum times */
            double tatmax = *last_t_value + tfall * trise * log(trise/tfall) / (trise - tfall);
            cm_analog_set_perm_bkpt(*last_t_value);
            cm_analog_set_perm_bkpt(tatmax);
        }
    }
    /* after initialization */
    else {
        /* individual scaling and delay */

        if (have_scaled) {
            /* */
            int i;
            allpulses = (pulse_info_t *) STATIC_VAR(pulses);
            for (i = 0; i < ports; i++){
                double tst = allpulses[i].start_time;
                if (tcurr < tst)
                    out = 0;
                else
                    out = allpulses[i].iscaled * (exp(-(tcurr-tst)/tfall) - exp(-(tcurr-tst)/trise));
                OUTPUT(out[i]) = out;
                OUTPUT(mon) = out;
            }
        }
        /* equal pulses, period, and repetition */
        else {
            last_t_value = (double *) STATIC_VAR(last_t_value);
            pulse_number = (int *) STATIC_VAR(pulse_number);
            last_ctrl = (double *) STATIC_VAR(last_ctrl);

            /* reset the pulse sequence, to start anew upon a rising ctrl */
            if (*last_ctrl < ctrlthres && ctrl >= ctrlthres) {
                *last_t_value = tcurr + tdelay;
                *pulse_number = 1;
            }
            *last_ctrl = ctrl;

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
                cm_message_printf("port name: out, node pair no.: %d, \nnode names: %s, %s, pulse time: %e",
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
            if (*pulse_number - 1 < PORT_SIZE(out)) {
                OUTPUT(out[*pulse_number - 1]) = out;
                OUTPUT(mon) = out;
            }
        }
    }
}




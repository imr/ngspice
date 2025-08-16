/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE degmonitor/cfunc.mod

Public Domain

Universty Duisburg-Essen
Duisburg, Germany
Project Flowspace

AUTHORS                      

    15 Aug 2025 Holger Vogt


MODIFICATIONS   



SUMMARY

    This file contains the model-specific routines used to
    functionally describe degradation monitor code model.


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
cm_degmon_callback(ARGS, Mif_Callback_Reason_t reason)
{
    switch (reason) {
        case MIF_CB_DESTROY: {
            double *constfac = STATIC_VAR (constfac);
            if (constfac)
                free(constfac);
            STATIC_VAR (constfac) = NULL;
            double *sintegral = STATIC_VAR (sintegral);
            if (sintegral)
                free(sintegral);
            STATIC_VAR (sintegral) = NULL;
            break;
        }
    }
}

           
/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/




                   
/*==============================================================================

FUNCTION void cm_seegen()

AUTHORS                      

    15 Aug 2025 Holger Vogt

SUMMARY

    This function implements the degradation monitor code model.

INTERFACES       

    FILE                 ROUTINE CALLED     

RETURNED VALUE
    
    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    model source: 
    IIS EAS, 2025


==============================================================================*/

/*=== CM_SEEGEN ROUTINE ===*/

void cm_degmon(ARGS)  /* structure holding parms, 
                                       inputs, outputs, etc.     */
{
    double vd;            /* drain voltage */
    double vg;            /* gate voltage */
    double vs;            /* source voltage */
    double vb;            /* bulk voltage */
    double A;             /* degradation model parameter */
    double Ea;            /* degradation model parameter */
    double b;             /* degradation model parameter */
    double L1;            /* degradation model parameter */
    double L2;            /* degradation model parameter */
    double n;             /* degradation model parameter */
    double c;             /* degradation model parameter */
    double L;             /* channel length */
    double *constfac;     /* static storage of const factor in model equation */
    double tfut;
    double tsim;
    double deg;           /* monitor output */
    double sintegrand = 0;
    double *sintegral;
    double *prevtime;
    double k = 1.38062259e-5; /* Boltzmann */


    if (ANALYSIS == MIF_AC) {
        return;
    }


    /* Retrieve frequently used parameters... */

    A = PARAM(A);
    Ea = PARAM(Ea);
    b = PARAM(b);
    L1 = PARAM(L1);
    L2 = PARAM(L2);
    n = PARAM(n);
    b = PARAM(b);
    c = PARAM(c);
    tfut = PARAM(tfuture);
    L = PARAM(L);
    tsim = TSTOP;


    if (INIT==1) {

        double Temp = TEMPERATURE + 273.15;
        
        if (PORT_SIZE(nodes) != 4)
        {
            cm_message_send("Error: only devices with exactly 4 node are currently supported\n");
            cm_cexit(1);
        }

        CALLBACK = cm_degmon_callback;

        /* Allocate storage for static values */
        STATIC_VAR(constfac) = (double *) malloc(sizeof(double));
        constfac = (double *) STATIC_VAR(constfac);
        *constfac = c * A * exp(Ea / k / Temp) * (L1 + pow((1 / L / 1e6) , L2));

        STATIC_VAR(sintegral) = (double *) malloc(sizeof(double));
        sintegral = (double *) STATIC_VAR(sintegral);
        *sintegral = 0.;

        STATIC_VAR(prevtime) = (double *) malloc(sizeof(double));
        prevtime = (double *) STATIC_VAR(prevtime);
        *prevtime = 0.;

    }
    else {

        if (ANALYSIS == MIF_DC) {
            return;
        }

        double x1, x2;

        constfac = (double *) STATIC_VAR(constfac);
        sintegral = (double *) STATIC_VAR(sintegral);
        prevtime = (double *) STATIC_VAR(prevtime);

        /* final time step quasi reached */
        if (*sintegral > 1e90) {
            return;
        }

        vd = INPUT(nodes[0]);
        vg = INPUT(nodes[1]);
        vs = INPUT(nodes[2]);
        vb = INPUT(nodes[3]);

        if (vd - vs > 0 && *prevtime < T(0)) {
            /**** model equations 1 ****/
            x1 = 1. / (*constfac * exp (b / (vd - vs)));
            x2 = -1. / n;
            sintegrand = pow(x1 , x2);
            *sintegral = *sintegral + sintegrand * (T(0) - T(1));
            /***************************/
            *prevtime = T(0);
        }

        /* test output */
        OUTPUT(mon) = *sintegral;

        if (T(0) > 0.99999 * tsim) {
            /**** model equations 2 ****/
            *sintegral = *sintegral * tfut / tsim;
            deg = 1. / (c * (pow(*sintegral, -1.* n)));
            /***************************/
            cm_message_printf("Degradation deg = %e\n", deg);
            *sintegral = 1e99; // flag final time step
        }
    }
}

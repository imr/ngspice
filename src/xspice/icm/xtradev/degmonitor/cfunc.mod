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
    functionally describe a degradation monitor code model.
    The model has been provided by IIS/EAS, Dresden, Germany,
    extracted for the IHP Open PDK and its MOS devoces modelled by PSP.


INTERFACES       

    FILE                 ROUTINE CALLED     


REFERENCED FILES

    Inputs from and outputs to ARGS structure.
                     

NON-STANDARD FEATURES

    NONE

===============================================================================*/

/*=== INCLUDE FILES ====================*/


#include "ngspice/mifdefs.h"
#include "ngspice/ngspice.h"
#include "ngspice/hash.h"
#include "ngspice/stringutil.h"
#include <string.h>

/*=== CONSTANTS ========================*/

/* maximum number of model parameters */
#define DEGPARAMAX 64
/* maximum number of models */
#define DEGMODMAX 64


/*=== MACROS ===========================*/



  
/*=== LOCAL VARIABLES & TYPEDEFS =======*/

struct agemod {
    char* devmodel;
    char* simmodel;
    int numparams;
    char *paramnames[DEGPARAMAX];
    char *paramvalstr[DEGPARAMAX];
    double paramvals[DEGPARAMAX];
    bool paramread[DEGPARAMAX];
    NGHASHPTR paramhash;
} *agemodptr;

/* This struct is model-specific: We need three data sets for dlt_vth [0],
   d_idlin [1], and d_idsat [2] */
typedef struct {
    double    constfac[3];   /* intermediate factor */
    double    sintegral[3];  /* intermediate intgral */
    double    prevtime[3];   /* previous time */
    int devtype;             /* device type 1: nms, -1: pmos */
    double VGS0;             /* degradation model parameter */
    double A[3];             /* degradation model parameter */
    double Ea[3];            /* degradation model parameter */
    double b[3];             /* degradation model parameter */
    double L1[3];            /* degradation model parameter */
    double L2[3];            /* degradation model parameter */
    double n[3];             /* degradation model parameter */
    double c[3];             /* degradation model parameter */
} degLocal_Data_t;



           
/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/

static void
cm_degmon_callback(ARGS, Mif_Callback_Reason_t reason)
{
    switch (reason) {
        case MIF_CB_DESTROY: {
            degLocal_Data_t *loc = (degLocal_Data_t *) STATIC_VAR(locdata);
            if (loc == (degLocal_Data_t *) NULL) {
                break;
            }

            free(loc);

            STATIC_VAR(locdata) = NULL;
            break;
        }
    }
}

/* use agemods as input to set the agemodel model parameters,
   e.g. loc->A[3] */
int
getdata(struct agemod *agemodptr, degLocal_Data_t *loc, char *devmod)
{
    int no; /* which device model */
    int ii, jj;

    /* all parameters lower case!
       At most 64 parameters are supported! */
    static char *names[] = {
        "vgs0",
        "a_dlt_vth",
        "ea_dlt_vth",
        "b_dlt_vth",
        "c_dlt_vth",
        "n_dlt_vth",
        "l1_dlt_vth",
        "l2_dlt_vth",
        "a_d_idlin",
        "ea_d_idlin",
        "b_d_idlin",
        "c_d_idlin",
        "n_d_idlin",
        "l1_d_idlin",
        "l2_d_idlin",
        "a_d_idsat",
        "ea_d_idsat",
        "b_d_idsat",
        "c_d_idsat",
        "n_d_idsat",
        "l1_d_idsat",
        "l2_d_idsat"
    };
/*
    static char *names[] = {
        "VGS0",
        "A_dlt_vth",
        "Ea_dlt_vth",
        "B_dlt_vth",
        "C_dlt_vth",
        "n_dlt_vth",
        "L1_dlt_vth",
        "L2_dlt_vth",
        "A_d_idlin",
        "Ea_d_idlin",
        "B_d_idlin",
        "C_d_idlin",
        "n_d_idlin",
        "L1_d_idlin",
        "L2_d_idlin",
        "A_d_idsat",
        "Ea_d_idsat",
        "B_d_idsat",
        "C_d_idsat",
        "n_d_idsat",
        "L1_d_idsat",
        "L2_d_idsat"
    };
*/
    double pvals[64];


    for (no = 0; no < 64; no++) {
       if (!agemodptr[no].devmodel){
           cm_message_printf("Error: Could not find device model %s in the degradation model data!\n", devmod);
           cm_cexit(1);
       }
       if (!strcasecmp(agemodptr[no].devmodel, devmod))
           break;
    }
    if (no == 64){
        cm_message_printf("Error: Could not find device model %s in the degradation model data!\n", devmod);
        cm_cexit(1);
    }

    /* Retrive model parameters from hash table. Sequence in pvals[i] is as set by names[i] above. */
    for (ii = 0; ii < agemodptr[no].numparams; ii++) {
        double* fval;
        fval = (double*)nghash_find(agemodptr[no].paramhash, names[ii]);
        if (fval){
            pvals[ii] = *fval;
        }
        else { /* error */
            cm_message_printf("Error: Could not retrieve parameter %s from the degradation model data!\n", names[ii]);
            cm_cexit(1);
        }
    }
    /* loc->xxx selected according to the sequence given in names[i]. */
    loc->VGS0 = pvals[0];
    loc->A[0] = pvals[1];
    loc->Ea[0] = pvals[2];
    loc->b[0] = pvals[3];
    loc->c[0] = pvals[4];
    loc->n[0] = pvals[5];
    loc->L1[0] = pvals[6];
    loc->L2[0] = pvals[7];
    loc->A[1] = pvals[8];
    loc->Ea[1] = pvals[9];
    loc->b[1] = pvals[10];
    loc->c[1] = pvals[11];
    loc->n[1] = pvals[12];
    loc->L1[1] = pvals[13];
    loc->L2[1] = pvals[14];
    loc->A[2] = pvals[15];
    loc->Ea[2] = pvals[16];
    loc->b[2] = pvals[17];
    loc->c[2] = pvals[18];
    loc->n[2] = pvals[19];
    loc->L1[2] = pvals[20];
    loc->L2[2] = pvals[21];
    return no;

}

                   
/*==============================================================================

FUNCTION void cm_degmon()

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
    double L;             /* channel length */
    double constfac;     /* static storage of const factor in model equation */
    double tfut;
    double tsim;
    double deg;           /* monitor output */
    double sintegrand = 0;
    double sintegral;
    double prevtime;
    double k = 1.38062259e-5; /* Boltzmann */
    int devtype;

    char *devmod;     /* PSP device model */
    char *simmod;     /* degradation model */

    double A;             /* degradation model parameter */
    double Ea;            /* degradation model parameter */
    double b;             /* degradation model parameter */
    double L1;            /* degradation model parameter */
    double L2;            /* degradation model parameter */
    double n;             /* degradation model parameter */
    double c;             /* degradation model parameter */

    degLocal_Data_t *loc;        /* Pointer to local static data, not to be included
                                       in the state vector */

    if (ANALYSIS == MIF_AC) {
        return;
    }

    /* Retrieve frequently used parameters... */

    devmod = PARAM(devmod);
    tfut = PARAM(tfuture);
    L = PARAM(L);
    tsim = TSTOP;

    if (INIT==1) {

        double Temp = TEMPERATURE + 273.15;
        int err = -1, ii;
        struct agemod *agemods = cm_get_deg_params();

        if (!agemods || !agemods[0].devmodel) {
            cm_message_send("Error: Could not find the degradation model data!\n");
            cm_cexit(1);
        }

        if (PORT_SIZE(nodes) != 4)
        {
            cm_message_send("Error: only devices with exactly 4 node are currently supported\n");
            cm_cexit(1);
        }

        CALLBACK = cm_degmon_callback;

        /*** allocate static storage for *loc ***/
        if ((loc = (degLocal_Data_t *) (STATIC_VAR(locdata) = calloc(1,
                sizeof(degLocal_Data_t)))) == (degLocal_Data_t *) NULL) {
            cm_message_send("Error: Unable to allocate Local_Data_t "
                "in code model degmon");
            cm_cexit(1);
        }

        /* use agemods as input to set the agemodel model parameters,
           e.g. loc->A[3] */
        err = getdata(agemods, loc, devmod);
        if (err == -1){
            cm_message_send("Error: Could not retrieve the degradation model info!\n");
            cm_cexit(1);
        }

        for (ii=0; ii < 3; ii++) {
            loc->constfac[ii] = loc->c[ii] * loc->A[ii] * exp(loc->Ea[ii] / k / Temp) 
                * (loc->L1[ii] + pow((1 / L / 1e6) , loc->L2[ii]));
            loc->sintegral[ii] = 0.;
            loc->prevtime[ii] = 0.;
        }

        if (strstr(devmod, "_nmos"))
            loc->devtype = 1;
        else if (strstr(devmod, "_pmos"))
            loc->devtype = -1;
        else {
            loc->devtype = 0;
            cm_message_send("Error: could not extract device type from model name\n");
            cm_cexit(1);
        }
/*
        cm_message_send(INSTNAME);
        cm_message_send(INSTMODNAME);
*/
    }
    else {

        int ii;

        if (ANALYSIS == MIF_DC) {
            return;
        }

        /* retrieve previous values */
        loc = STATIC_VAR (locdata);

        if (!loc)
            return;

        vd = INPUT(nodes[0]);
        vg = INPUT(nodes[1]);
        vs = INPUT(nodes[2]);
        vb = INPUT(nodes[3]);

        devtype = loc->devtype;

        if (devtype == -1){
            vd *= -1.;
            vg *= -1.;
            vs *= -1.;
            vb *= -1.;
        }


        for (ii = 0; ii < 3; ii++) {
            double x1, x2;
            constfac = loc->constfac[ii];
            sintegral = loc->sintegral[ii];
            prevtime = loc->prevtime[ii];
            b = loc->b[ii];
            n = loc->n[ii];
            c = loc->c[ii];

            /* final time step quasi reached */
            if (sintegral > 1e90) {
                return;
            }

            if (vd - vs > 0 && prevtime < T(0)) {
                /**** model equations 1 ****/
                x1 = 1. / (constfac * exp (b / (vd - vs)));
                x2 = -1. / n;
                sintegrand = pow(x1 , x2);
                sintegral = sintegral + sintegrand * (T(0) - T(1));
                /***************************/
                prevtime = T(0);
            }

            /* test output */
            if(ii == 0)
                OUTPUT(mon) = sintegral;

            if (T(0) > 0.99999 * tsim) {
                /**** model equations 2 ****/
                sintegral = sintegral * tfut / tsim;
                deg = 1. / (c * (pow(sintegral, -1.* n)));
                /***************************/
                cm_message_printf("no. %d, Degradation deg = %e\n", ii, deg);
                sintegral = 1e99; // flag final time step
            }
            loc->sintegral[ii] = sintegral;
            loc->prevtime[ii] = prevtime;
        }
    }
}

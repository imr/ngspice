/*============================================================================
FILE    MIFtrunc.c

MEMBER OF process XSPICE

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503

AUTHORS

    9/12/91  Bill Kuhn

MODIFICATIONS

    <date> <person name> <nature of modifications>

SUMMARY

    This file contains the function called by SPICE to check truncation
    error of an integration state used by a code model.

INTERFACES

    MIFtrunc()

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/

/* #include "prefix.h" */
#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/cktdefs.h"
#include "ngspice/sperror.h"

//#include "util.h"
#include <math.h>

#include "ngspice/mifproto.h"
#include "ngspice/mifparse.h"
#include "ngspice/mifdefs.h"
#include "ngspice/mifcmdat.h"

/* #include "suffix.h" */



static void MIFterr(Mif_Intgr_t *intgr, CKTcircuit *ckt, double *timeStep);



/*
MIFtrunc

This function is called by the CKTtrunc() driver function to
check numerical integration truncation error of any integrals
associated with instances of a particular code model type.  It
traverses all models of that type and all instances of each
model.  For each instance, it looks in the instance structure to
determine if any variables allocated by cm_analog_alloc() have been used
in a call to cm_analog_integrate().  If so, the truncation error of that
integration is computed and used to set the maximum delta allowed
for the current timestep.
*/


int
MIFtrunc(
    GENmodel   *inModel,    /* The head of the model list */
    CKTcircuit *ckt,        /* The circuit structure */
    double     *timeStep)   /* The timestep delta */
{

    MIFmodel    *model;
    MIFinstance *here;

    int         i;


    /* Setup for access into MIF specific model data */
    model = (MIFmodel *) inModel;


    /* loop through all models of this type */
    for( ; model != NULL; model = MIFnextModel(model)) {

        /* Loop through all instances of this model */
        for(here = MIFinstances(model); here != NULL; here = MIFnextInstance(here)) {

            /* Loop through all integration states on the instance */
            for(i = 0; i < here->num_intgr; i++) {

                /* Limit timeStep according to truncation error */
                MIFterr(&(here->intgr[i]), ckt, timeStep);

            } /* end for number of integration states */
        } /* end for all instances */
    }  /* end for all models of this type */


    return(OK);
}



/*
 *
 * This is a modified version of the function CKTterr().  It limits
 * timeStep according to computed truncation error.
 *
 * Modifications are Copyright 1991 Georgia Tech Research Institute
 *
 */


static void MIFterr(
    Mif_Intgr_t *intgr,
    CKTcircuit  *ckt,
    double      *timeStep)
{
    double volttol;
    double chargetol;
    double tol;
    double del;
    double diff[8];
    double deltmp[8];
    double factor;

    int i;
    int j;

    static double gearCoeff[] = {
        .5,
        .2222222222,
        .1363636364,
        .096,
        .07299270073,
        .05830903790
    };
    static double trapCoeff[] = {
        .5,
        .08333333333
    };

    /* Define new local variables. Dimension = number of states in ckt struct */
    char        *byte_aligned_state_ptr;
    double      *state_ptr[8];


    /* Set state pointers to the (possibly byte-aligned) states */
    for(i = 0; i < 8; i++) {
        byte_aligned_state_ptr =  (char *) ckt->CKTstates[i];
        byte_aligned_state_ptr += intgr->byte_index;
        state_ptr[i] = (double *) byte_aligned_state_ptr;
    }

    /* Modify computation of volttol to not include current from previous timestep */
    /* which is unavailable in this implementation. Note that this makes the       */
    /* the overall trunction error timestep smaller (which is better accuracy)     */

    /* Old code */
/*
    volttol = ckt->CKTabstol + ckt->CKTreltol *
            MAX( fabs(ckt->CKTstate0[ccap]), fabs(ckt->CKTstate1[ccap]));
*/

    /* New code */
    volttol = ckt->CKTabstol + ckt->CKTreltol * fabs(*(state_ptr[0]) - *(state_ptr[1]))
                                              / ckt->CKTdelta;

    /* Modify remaining references to qcap to access byte-aligned MIF state */
    /* Otherwise, remaining code is same as SPICE3C1 ... */

    chargetol = MAX(fabs(*(state_ptr[0])),fabs(*(state_ptr[1])));
    chargetol = ckt->CKTreltol * MAX(chargetol,ckt->CKTchgtol)/ckt->CKTdelta;
    tol = MAX(volttol,chargetol);
    /* now divided differences */
    for(i=ckt->CKTorder+1;i>=0;i--) {
        diff[i] = *(state_ptr[i]);
    }
    for(i=0 ; i <= ckt->CKTorder ; i++) {
        deltmp[i] = ckt->CKTdeltaOld[i];
    }
    j = ckt->CKTorder;
    for (;;) {
        for(i=0;i <= j;i++) {
            diff[i] = (diff[i] - diff[i+1])/deltmp[i];
        }
        if (--j < 0) break;
        for(i=0;i <= j;i++) {
            deltmp[i] = deltmp[i+1] + ckt->CKTdeltaOld[i];
        }
    }
    switch(ckt->CKTintegrateMethod) {
    case GEAR:
    default:
            factor = gearCoeff[ckt->CKTorder-1];
            break;

    case TRAPEZOIDAL:
            factor = trapCoeff[ckt->CKTorder - 1] ;
            break;
    }
    del = ckt->CKTtrtol * tol/MAX(ckt->CKTabstol,factor * fabs(diff[0]));
    if(ckt->CKTorder == 2) {
        del = sqrt(del);
    } else if (ckt->CKTorder > 2) {
        del = exp(log(del)/ckt->CKTorder);
    }
    *timeStep = MIN(*timeStep,del);
    return;

}

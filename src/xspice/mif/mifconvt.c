/*============================================================================
FILE    MIFconvTest.c

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

    This file contains the function used to check that internal
    states of a code model have converged.  These internal states
    are typically integration states.

INTERFACES

    MIFconvTest()

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
#include "ngspice/devdefs.h"
//#include "CONST.h"
#include "ngspice/trandefs.h"
#include <math.h>

#include "ngspice/enh.h"

#include "ngspice/mifproto.h"
#include "ngspice/mifparse.h"
#include "ngspice/mifdefs.h"
#include "ngspice/mifcmdat.h"

/* #include "suffix.h"  */




/*
MIFconvTest

This function is called by the CKTconvTest() driver function to
check convergence of any states owned by instances of a
particular code model type.  It loops through all models of that
type and all instances of each model.  For each instance, it
looks in the instance structure to determine if any variables
allocated by cm_analog_alloc() have been registered by a call to
cm_analog_converge() to have their convergence tested.  If so, the value
of the function at the last iteration is compared with the value
at the current iteration to see if it has converged to within the
same delta amount used in node convergence checks (as defined by
SPICE 3C1).
*/


int MIFconvTest(
    GENmodel   *inModel,   /* The head of the model list */
    CKTcircuit *ckt)       /* The circuit structure */
{

    MIFmodel    *model;
    MIFinstance *here;

    int         i;

    double      value;
    double      last_value;

    char        *byte_aligned_double_ptr;
    double      *double_ptr;

    double      tol;

    Mif_Boolean_t  gotone = MIF_FALSE;


    /* Setup for access into MIF specific model data */
    model = (MIFmodel *) inModel;

    /* loop through all models of this type */
    for( ; model != NULL; model = MIFnextModel(model)) {

        /* Loop through all instances of this model */
        for(here = MIFinstances(model); here != NULL; here = MIFnextInstance(here)) {

            /* Loop through all items registered for convergence */
            for(i = 0; i < here->num_conv; i++) {

                /* Get the current value and the last value */
                byte_aligned_double_ptr = (char *) ckt->CKTstate0;
                byte_aligned_double_ptr += here->conv[i].byte_index;
                double_ptr = (double *) byte_aligned_double_ptr;
                value = *double_ptr;

                last_value = here->conv[i].last_value;

                /* If none have failed so far, check convergence */
                if(! gotone) {

                    tol = ckt->CKTreltol * MAX(fabs(value), fabs(last_value))
                                         + ckt->CKTabstol;
                    if (fabs(value - last_value) > tol) {
                        if(ckt->enh->conv_debug.report_conv_probs) {
                            ENHreport_conv_prob(ENH_ANALOG_INSTANCE,
                                                here->MIFname,
                                                "");
                        }
                        ckt->CKTnoncon++;
                        gotone = MIF_TRUE;
                    }
                }

                /* Rotate the current value to last_value */
                here->conv[i].last_value = value;

            } /* end for number of conv items */
        } /* end for all instances */
    }  /* end for all models of this type */

    return(OK);
}

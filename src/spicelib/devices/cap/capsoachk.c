/**********
Copyright 2013 Dietmar Warning. All rights reserved.
Author:   2013 Dietmar Warning
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "capdefs.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/cpdefs.h"


int
CAPsoaCheck(CKTcircuit *ckt, GENmodel *inModel)
{
    CAPmodel *model = (CAPmodel *) inModel;
    CAPinstance *here;
    double vc;  /* current capacitor voltage */
    int maxwarns;
    static int warns_bv = 0;

    if (!ckt) {
        warns_bv = 0;
        return OK;
    }

    maxwarns = ckt->CKTsoaMaxWarns;

    for (; model; model = CAPnextModel(model)) {

        for (here = CAPinstances(model); here; here = CAPnextInstance(here)) {

            vc = fabs(ckt->CKTrhsOld [here->CAPposNode] -
                      ckt->CKTrhsOld [here->CAPnegNode]);

            if (vc > here->CAPbv_max)
                if (warns_bv < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vc|=%g has exceeded Bv_max=%g\n",
                               vc, here->CAPbv_max);
                    warns_bv++;
                }

        }

    }

    return OK;
}

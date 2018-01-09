/**********
Copyright 2013 Dietmar Warning. All rights reserved.
Author:   2013 Dietmar Warning
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "resdefs.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/cpdefs.h"


int
RESsoaCheck(CKTcircuit *ckt, GENmodel *inModel)
{
    RESmodel *model = (RESmodel *) inModel;
    RESinstance *here;
    double vr;  /* current resistor voltage */
    int maxwarns;
    static int warns_bv = 0;

    if (!ckt) {
        warns_bv = 0;
        return OK;
    }

    maxwarns = ckt->CKTsoaMaxWarns;

    for (; model; model = RESnextModel(model)) {

        for (here = RESinstances(model); here; here = RESnextInstance(here)) {

            vr = fabs(ckt->CKTrhsOld [here->RESposNode] -
                      ckt->CKTrhsOld [here->RESnegNode]);

            if (vr > here->RESbv_max)
                if (warns_bv < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vr|=%g has exceeded Bv_max=%g\n",
                               vr, here->RESbv_max);
                    warns_bv++;
                }

        }

    }

    return OK;
}

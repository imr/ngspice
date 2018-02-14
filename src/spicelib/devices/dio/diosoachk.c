/**********
Copyright 2013 Dietmar Warning. All rights reserved.
Author:   2013 Dietmar Warning
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "diodefs.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/cpdefs.h"


int
DIOsoaCheck(CKTcircuit *ckt, GENmodel *inModel)
{
    DIOmodel *model = (DIOmodel *) inModel;
    DIOinstance *here;
    double vd;  /* current diode voltage */
    int maxwarns;
    static int warns_fv = 0, warns_bv = 0;

    if (!ckt) {
        warns_fv = 0;
        warns_bv = 0;
        return OK;
    }

    maxwarns = ckt->CKTsoaMaxWarns;

    for (; model; model = DIOnextModel(model)) {

        for (here = DIOinstances(model); here; here = DIOnextInstance(here)) {

            vd = ckt->CKTrhsOld [here->DIOposPrimeNode] -
                 ckt->CKTrhsOld [here->DIOnegNode];

            if (vd > model->DIOfv_max)
                if (warns_fv < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "Vj=%g has exceeded Fv_max=%g\n",
                               vd, model->DIOfv_max);
                    warns_fv++;
                }

            if (-vd > model->DIObv_max)
                if (warns_bv < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "Vj=%g has exceeded Bv_max=%g\n",
                               vd, model->DIObv_max);
                    warns_bv++;
                }

        }

    }

    return OK;
}

/**********
Copyright 2013 Dietmar Warning. All rights reserved.
Author:   2013 Dietmar Warning
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vbicdefs.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/cpdefs.h"


int
VBICsoaCheck(CKTcircuit *ckt, GENmodel *inModel)
{
    VBICmodel *model = (VBICmodel *) inModel;
    VBICinstance *here;
    double vbe, vbc, vce;    /* actual bjt voltages */
    int maxwarns;
    static int warns_vbe = 0, warns_vbc = 0, warns_vce = 0;

    if (!ckt) {
        warns_vbe = 0;
        warns_vbc = 0;
        warns_vce = 0;
        return OK;
    }

    maxwarns = ckt->CKTsoaMaxWarns;

    for (; model; model = VBICnextModel(model)) {

        for (here = VBICinstances(model); here; here=VBICnextInstance(here)) {

            vbe = fabs(ckt->CKTrhsOld [here->VBICbaseNode] -
                       ckt->CKTrhsOld [here->VBICemitNode]);
            vbc = fabs(ckt->CKTrhsOld [here->VBICbaseNode] -
                       ckt->CKTrhsOld [here->VBICcollNode]);
            vce = fabs(ckt->CKTrhsOld [here->VBICcollNode] -
                       ckt->CKTrhsOld [here->VBICemitNode]);

            if (vbe > model->VBICvbeMax)
                if (warns_vbe < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vbe|=%g has exceeded Vbe_max=%g\n",
                               vbe, model->VBICvbeMax);
                    warns_vbe++;
                }

            if (vbc > model->VBICvbcMax)
                if (warns_vbc < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vbc|=%g has exceeded Vbc_max=%g\n",
                               vbc, model->VBICvbcMax);
                    warns_vbc++;
                }

            if (vce > model->VBICvceMax)
                if (warns_vce < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vce|=%g has exceeded Vce_max=%g\n",
                               vce, model->VBICvceMax);
                    warns_vce++;
                }

        }
    }

    return OK;
}

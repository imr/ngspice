/**********
Copyright 2013 Dietmar Warning. All rights reserved.
Author:   2013 Dietmar Warning
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bjtdefs.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/cpdefs.h"


int
BJTsoaCheck(CKTcircuit *ckt, GENmodel *inModel)
{
    BJTmodel *model = (BJTmodel *) inModel;
    BJTinstance *here;
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

    for (; model; model = BJTnextModel(model)) {

        for (here = BJTinstances(model); here; here=BJTnextInstance(here)) {

            vbe = fabs(ckt->CKTrhsOld [here->BJTbasePrimeNode] -
                       ckt->CKTrhsOld [here->BJTemitPrimeNode]);
            vbc = fabs(ckt->CKTrhsOld [here->BJTbasePrimeNode] -
                       ckt->CKTrhsOld [here->BJTcolPrimeNode]);
            vce = fabs(ckt->CKTrhsOld [here->BJTcolPrimeNode] -
                       ckt->CKTrhsOld [here->BJTemitPrimeNode]);

            if (vbe > model->BJTvbeMax)
                if (warns_vbe < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vbe|=%g has exceeded Vbe_max=%g\n",
                               vbe, model->BJTvbeMax);
                    warns_vbe++;
                }

            if (vbc > model->BJTvbcMax)
                if (warns_vbc < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vbc|=%g has exceeded Vbc_max=%g\n",
                               vbc, model->BJTvbcMax);
                    warns_vbc++;
                }

            if (vce > model->BJTvceMax)
                if (warns_vce < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vce|=%g has exceeded Vce_max=%g\n",
                               vce, model->BJTvceMax);
                    warns_vce++;
                }

        }
    }

    return OK;
}

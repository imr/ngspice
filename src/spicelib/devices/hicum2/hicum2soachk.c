/**********
License              : 3-clause BSD
Spice3 Implementation: 2019-2020 Dietmar Warning, Markus Müller, Mario Krattenmacher
Model Author         : 1990 Michael Schröter TU Dresden
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hicum2defs.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/cpdefs.h"


int
HICUMsoaCheck(CKTcircuit *ckt, GENmodel *inModel)
{
    HICUMmodel *model = (HICUMmodel *) inModel;
    HICUMinstance *here;
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

    for (; model; model = HICUMnextModel(model)) {

        for (here = HICUMinstances(model); here; here=HICUMnextInstance(here)) {

            vbe = fabs(ckt->CKTrhsOld [here->HICUMbaseNode] -
                       ckt->CKTrhsOld [here->HICUMemitNode]);
            vbc = fabs(ckt->CKTrhsOld [here->HICUMbaseNode] -
                       ckt->CKTrhsOld [here->HICUMcollNode]);
            vce = fabs(ckt->CKTrhsOld [here->HICUMcollNode] -
                       ckt->CKTrhsOld [here->HICUMemitNode]);

            if (vbe > model->HICUMvbeMax)
                if (warns_vbe < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vbe|=%g has exceeded Vbe_max=%g\n",
                               vbe, model->HICUMvbeMax);
                    warns_vbe++;
                }

            if (vbc > model->HICUMvbcMax)
                if (warns_vbc < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vbc|=%g has exceeded Vbc_max=%g\n",
                               vbc, model->HICUMvbcMax);
                    warns_vbc++;
                }

            if (vce > model->HICUMvceMax)
                if (warns_vce < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vce|=%g has exceeded Vce_max=%g\n",
                               vce, model->HICUMvceMax);
                    warns_vce++;
                }

        }
    }

    return OK;
}

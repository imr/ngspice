/**********
Copyright 2013 Dietmar Warning. All rights reserved.
Author:   2013 Dietmar Warning
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b4soidef.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/cpdefs.h"


int
B4SOIsoaCheck(CKTcircuit *ckt, GENmodel *inModel)
{
    B4SOImodel *model = (B4SOImodel *) inModel;
    B4SOIinstance *here;
    double vgs, vgd, vgb, vds, vbs, vbd;    /* actual mos voltages */
    int maxwarns;
    static int warns_vgs = 0, warns_vgd = 0, warns_vgb = 0, warns_vds = 0, warns_vbs = 0, warns_vbd = 0;

    if (!ckt) {
        warns_vgs = 0;
        warns_vgd = 0;
        warns_vgb = 0;
        warns_vds = 0;
        warns_vbs = 0;
        warns_vbd = 0;
        return OK;
    }

    maxwarns = ckt->CKTsoaMaxWarns;

    for (; model; model = model->B4SOInextModel) {

        for (here = model->B4SOIinstances; here; here = here->B4SOInextInstance) {

            vgs = fabs(ckt->CKTrhsOld [here->B4SOIgNode] -
                       ckt->CKTrhsOld [here->B4SOIsNodePrime]);

            vgd = fabs(ckt->CKTrhsOld [here->B4SOIgNode] -
                       ckt->CKTrhsOld [here->B4SOIdNodePrime]);

            vgb = fabs(ckt->CKTrhsOld [here->B4SOIgNode] -
                       ckt->CKTrhsOld [here->B4SOIbNode]);

            vds = fabs(ckt->CKTrhsOld [here->B4SOIdNodePrime] -
                       ckt->CKTrhsOld [here->B4SOIsNodePrime]);

            vbs = fabs(ckt->CKTrhsOld [here->B4SOIbNode] -
                       ckt->CKTrhsOld [here->B4SOIsNodePrime]);

            vbd = fabs(ckt->CKTrhsOld [here->B4SOIbNode] -
                       ckt->CKTrhsOld [here->B4SOIdNodePrime]);

            if (vgs > model->B4SOIvgsMax)
                if (warns_vgs < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vgs|=%g has exceeded Vgs_max=%g\n",
                               vgs, model->B4SOIvgsMax);
                    warns_vgs++;
                }

            if (vgd > model->B4SOIvgdMax)
                if (warns_vgd < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vgd|=%g has exceeded Vgd_max=%g\n",
                               vgd, model->B4SOIvgdMax);
                    warns_vgd++;
                }

            if (vgb > model->B4SOIvgbMax)
                if (warns_vgb < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vgb|=%g has exceeded Vgb_max=%g\n",
                               vgb, model->B4SOIvgbMax);
                    warns_vgb++;
                }

            if (vds > model->B4SOIvdsMax)
                if (warns_vds < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vds|=%g has exceeded Vds_max=%g\n",
                               vds, model->B4SOIvdsMax);
                    warns_vds++;
                }

            if (vbs > model->B4SOIvbsMax)
                if (warns_vbs < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vbs|=%g has exceeded Vbs_max=%g\n",
                               vbs, model->B4SOIvbsMax);
                    warns_vbs++;
                }

            if (vbd > model->B4SOIvbdMax)
                if (warns_vbd < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vbd|=%g has exceeded Vbd_max=%g\n",
                               vbd, model->B4SOIvbdMax);
                    warns_vbd++;
                }

        }
    }

    return OK;
}

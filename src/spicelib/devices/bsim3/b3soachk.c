/**********
Copyright 2013 Dietmar Warning. All rights reserved.
Author:   2013 Dietmar Warning
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3def.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/cpdefs.h"


int
BSIM3soaCheck(CKTcircuit *ckt, GENmodel *inModel)
{
    BSIM3model *model = (BSIM3model *) inModel;
    BSIM3instance *here;
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

    for (; model; model = model->BSIM3nextModel) {

        for (here = model->BSIM3instances; here; here = here->BSIM3nextInstance) {

            vgs = fabs(ckt->CKTrhsOld [here->BSIM3gNode] -
                       ckt->CKTrhsOld [here->BSIM3sNodePrime]);

            vgd = fabs(ckt->CKTrhsOld [here->BSIM3gNode] -
                       ckt->CKTrhsOld [here->BSIM3dNodePrime]);

            vgb = fabs(ckt->CKTrhsOld [here->BSIM3gNode] -
                       ckt->CKTrhsOld [here->BSIM3bNode]);

            vds = fabs(ckt->CKTrhsOld [here->BSIM3dNodePrime] -
                       ckt->CKTrhsOld [here->BSIM3sNodePrime]);

            vbs = fabs(ckt->CKTrhsOld [here->BSIM3bNode] -
                       ckt->CKTrhsOld [here->BSIM3sNodePrime]);

            vbd = fabs(ckt->CKTrhsOld [here->BSIM3bNode] -
                       ckt->CKTrhsOld [here->BSIM3dNodePrime]);

            if (vgs > model->BSIM3vgsMax)
                if (warns_vgs < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vgs|=%g has exceeded Vgs_max=%g\n",
                               vgs, model->BSIM3vgsMax);
                    warns_vgs++;
                }

            if (vgd > model->BSIM3vgdMax)
                if (warns_vgd < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vgd|=%g has exceeded Vgd_max=%g\n",
                               vgd, model->BSIM3vgdMax);
                    warns_vgd++;
                }

            if (vgb > model->BSIM3vgbMax)
                if (warns_vgb < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vgb|=%g has exceeded Vgb_max=%g\n",
                               vgb, model->BSIM3vgbMax);
                    warns_vgb++;
                }

            if (vds > model->BSIM3vdsMax)
                if (warns_vds < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vds|=%g has exceeded Vds_max=%g\n",
                               vds, model->BSIM3vdsMax);
                    warns_vds++;
                }

            if (vbs > model->BSIM3vbsMax)
                if (warns_vbs < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vbs|=%g has exceeded Vbs_max=%g\n",
                               vbs, model->BSIM3vbsMax);
                    warns_vbs++;
                }

            if (vbd > model->BSIM3vbdMax)
                if (warns_vbd < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vbd|=%g has exceeded Vbd_max=%g\n",
                               vbd, model->BSIM3vbdMax);
                    warns_vbd++;
                }

        }
    }

    return OK;
}

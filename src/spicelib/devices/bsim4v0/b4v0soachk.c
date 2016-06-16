/**********
Copyright 2013 Dietmar Warning. All rights reserved.
Author:   2013 Dietmar Warning
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v0def.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/cpdefs.h"


int
BSIM4v0soaCheck(CKTcircuit *ckt, GENmodel *inModel)
{
    BSIM4v0model *model = (BSIM4v0model *) inModel;
    BSIM4v0instance *here;
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

    for (; model; model = model->BSIM4v0nextModel) {

        for (here = model->BSIM4v0instances; here; here = here->BSIM4v0nextInstance) {

            vgs = fabs(ckt->CKTrhsOld [here->BSIM4v0gNodePrime] -
                       ckt->CKTrhsOld [here->BSIM4v0sNodePrime]);

            vgd = fabs(ckt->CKTrhsOld [here->BSIM4v0gNodePrime] -
                       ckt->CKTrhsOld [here->BSIM4v0dNodePrime]);

            vgb = fabs(ckt->CKTrhsOld [here->BSIM4v0gNodePrime] -
                       ckt->CKTrhsOld [here->BSIM4v0bNodePrime]);

            vds = fabs(ckt->CKTrhsOld [here->BSIM4v0dNodePrime] -
                       ckt->CKTrhsOld [here->BSIM4v0sNodePrime]);

            vbs = fabs(ckt->CKTrhsOld [here->BSIM4v0bNodePrime] -
                       ckt->CKTrhsOld [here->BSIM4v0sNodePrime]);

            vbd = fabs(ckt->CKTrhsOld [here->BSIM4v0bNodePrime] -
                       ckt->CKTrhsOld [here->BSIM4v0dNodePrime]);

            if (vgs > model->BSIM4v0vgsMax)
                if (warns_vgs < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vgs|=%g has exceeded Vgs_max=%g\n",
                               vgs, model->BSIM4v0vgsMax);
                    warns_vgs++;
                }

            if (vgd > model->BSIM4v0vgdMax)
                if (warns_vgd < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vgd|=%g has exceeded Vgd_max=%g\n",
                               vgd, model->BSIM4v0vgdMax);
                    warns_vgd++;
                }

            if (vgb > model->BSIM4v0vgbMax)
                if (warns_vgb < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vgb|=%g has exceeded Vgb_max=%g\n",
                               vgb, model->BSIM4v0vgbMax);
                    warns_vgb++;
                }

            if (vds > model->BSIM4v0vdsMax)
                if (warns_vds < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vds|=%g has exceeded Vds_max=%g\n",
                               vds, model->BSIM4v0vdsMax);
                    warns_vds++;
                }

            if (vbs > model->BSIM4v0vbsMax)
                if (warns_vbs < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vbs|=%g has exceeded Vbs_max=%g\n",
                               vbs, model->BSIM4v0vbsMax);
                    warns_vbs++;
                }

            if (vbd > model->BSIM4v0vbdMax)
                if (warns_vbd < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vbd|=%g has exceeded Vbd_max=%g\n",
                               vbd, model->BSIM4v0vbdMax);
                    warns_vbd++;
                }

        }
    }

    return OK;
}

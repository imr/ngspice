/**********
Copyright 2013 Dietmar Warning. All rights reserved.
Author:   2013 Dietmar Warning
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3v32def.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/cpdefs.h"


int
BSIM3v32soaCheck(CKTcircuit *ckt, GENmodel *inModel)
{
    BSIM3v32model *model = (BSIM3v32model *) inModel;
    BSIM3v32instance *here;
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

    for (; model; model = model->BSIM3v32nextModel) {

        for (here = model->BSIM3v32instances; here; here = here->BSIM3v32nextInstance) {

            vgs = fabs(ckt->CKTrhsOld [here->BSIM3v32gNode] -
                       ckt->CKTrhsOld [here->BSIM3v32sNodePrime]);

            vgd = fabs(ckt->CKTrhsOld [here->BSIM3v32gNode] -
                       ckt->CKTrhsOld [here->BSIM3v32dNodePrime]);

            vgb = fabs(ckt->CKTrhsOld [here->BSIM3v32gNode] -
                       ckt->CKTrhsOld [here->BSIM3v32bNode]);

            vds = fabs(ckt->CKTrhsOld [here->BSIM3v32dNodePrime] -
                       ckt->CKTrhsOld [here->BSIM3v32sNodePrime]);

            vbs = fabs(ckt->CKTrhsOld [here->BSIM3v32bNode] -
                       ckt->CKTrhsOld [here->BSIM3v32sNodePrime]);

            vbd = fabs(ckt->CKTrhsOld [here->BSIM3v32bNode] -
                       ckt->CKTrhsOld [here->BSIM3v32dNodePrime]);

            if (vgs > model->BSIM3v32vgsMax)
                if (warns_vgs < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vgs|=%g has exceeded Vgs_max=%g\n",
                               vgs, model->BSIM3v32vgsMax);
                    warns_vgs++;
                }

            if (vgd > model->BSIM3v32vgdMax)
                if (warns_vgd < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vgd|=%g has exceeded Vgd_max=%g\n",
                               vgd, model->BSIM3v32vgdMax);
                    warns_vgd++;
                }

            if (vgb > model->BSIM3v32vgbMax)
                if (warns_vgb < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vgb|=%g has exceeded Vgb_max=%g\n",
                               vgb, model->BSIM3v32vgbMax);
                    warns_vgb++;
                }

            if (vds > model->BSIM3v32vdsMax)
                if (warns_vds < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vds|=%g has exceeded Vds_max=%g\n",
                               vds, model->BSIM3v32vdsMax);
                    warns_vds++;
                }

            if (vbs > model->BSIM3v32vbsMax)
                if (warns_vbs < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vbs|=%g has exceeded Vbs_max=%g\n",
                               vbs, model->BSIM3v32vbsMax);
                    warns_vbs++;
                }

            if (vbd > model->BSIM3v32vbdMax)
                if (warns_vbd < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vbd|=%g has exceeded Vbd_max=%g\n",
                               vbd, model->BSIM3v32vbdMax);
                    warns_vbd++;
                }

        }
    }

    return OK;
}

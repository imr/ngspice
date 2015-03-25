/**********
Copyright 2013 Dietmar Warning. All rights reserved.
Author:   2013 Dietmar Warning
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hsmhvdef.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/cpdefs.h"


int
HSMHV2soaCheck(CKTcircuit *ckt, GENmodel *inModel)
{
    HSMHV2model *model = (HSMHV2model *) inModel;
    HSMHV2instance *here;
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

    for (; model; model = model->HSMHV2nextModel) {

        for (here = model->HSMHV2instances; here; here = here->HSMHV2nextInstance) {

            vgs = fabs(ckt->CKTrhsOld [here->HSMHV2gNode] -
                       ckt->CKTrhsOld [here->HSMHV2sNodePrime]);

            vgd = fabs(ckt->CKTrhsOld [here->HSMHV2gNode] -
                       ckt->CKTrhsOld [here->HSMHV2dNodePrime]);

            vgb = fabs(ckt->CKTrhsOld [here->HSMHV2gNode] -
                       ckt->CKTrhsOld [here->HSMHV2bNode]);

            vds = fabs(ckt->CKTrhsOld [here->HSMHV2dNodePrime] -
                       ckt->CKTrhsOld [here->HSMHV2sNodePrime]);

            vbs = fabs(ckt->CKTrhsOld [here->HSMHV2bNode] -
                       ckt->CKTrhsOld [here->HSMHV2sNodePrime]);

            vbd = fabs(ckt->CKTrhsOld [here->HSMHV2bNode] -
                       ckt->CKTrhsOld [here->HSMHV2dNodePrime]);

            if (vgs > model->HSMHV2vgsMax)
                if (warns_vgs < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vgs|=%g has exceeded Vgs_max=%g\n",
                               vgs, model->HSMHV2vgsMax);
                    warns_vgs++;
                }

            if (vgd > model->HSMHV2vgdMax)
                if (warns_vgd < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vgd|=%g has exceeded Vgd_max=%g\n",
                               vgd, model->HSMHV2vgdMax);
                    warns_vgd++;
                }

            if (vgb > model->HSMHV2vgbMax)
                if (warns_vgb < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vgb|=%g has exceeded Vgb_max=%g\n",
                               vgb, model->HSMHV2vgbMax);
                    warns_vgb++;
                }

            if (vds > model->HSMHV2vdsMax)
                if (warns_vds < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vds|=%g has exceeded Vds_max=%g\n",
                               vds, model->HSMHV2vdsMax);
                    warns_vds++;
                }

            if (vbs > model->HSMHV2vbsMax)
                if (warns_vbs < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vbs|=%g has exceeded Vbs_max=%g\n",
                               vbs, model->HSMHV2vbsMax);
                    warns_vbs++;
                }

            if (vbd > model->HSMHV2vbdMax)
                if (warns_vbd < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vbd|=%g has exceeded Vbd_max=%g\n",
                               vbd, model->HSMHV2vbdMax);
                    warns_vbd++;
                }

        }
    }

    return OK;
}

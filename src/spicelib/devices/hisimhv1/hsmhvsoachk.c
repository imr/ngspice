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
HSMHVsoaCheck(CKTcircuit *ckt, GENmodel *inModel)
{
    HSMHVmodel *model = (HSMHVmodel *) inModel;
    HSMHVinstance *here;
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

    for (; model; model = model->HSMHVnextModel) {

        for (here = model->HSMHVinstances; here; here = here->HSMHVnextInstance) {

            vgs = fabs(ckt->CKTrhsOld [here->HSMHVgNode] -
                       ckt->CKTrhsOld [here->HSMHVsNodePrime]);

            vgd = fabs(ckt->CKTrhsOld [here->HSMHVgNode] -
                       ckt->CKTrhsOld [here->HSMHVdNodePrime]);

            vgb = fabs(ckt->CKTrhsOld [here->HSMHVgNode] -
                       ckt->CKTrhsOld [here->HSMHVbNode]);

            vds = fabs(ckt->CKTrhsOld [here->HSMHVdNodePrime] -
                       ckt->CKTrhsOld [here->HSMHVsNodePrime]);

            vbs = fabs(ckt->CKTrhsOld [here->HSMHVbNode] -
                       ckt->CKTrhsOld [here->HSMHVsNodePrime]);

            vbd = fabs(ckt->CKTrhsOld [here->HSMHVbNode] -
                       ckt->CKTrhsOld [here->HSMHVdNodePrime]);

            if (vgs > model->HSMHVvgsMax)
                if (warns_vgs < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vgs|=%g has exceeded Vgs_max=%g\n",
                               vgs, model->HSMHVvgsMax);
                    warns_vgs++;
                }

            if (vgd > model->HSMHVvgdMax)
                if (warns_vgd < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vgd|=%g has exceeded Vgd_max=%g\n",
                               vgd, model->HSMHVvgdMax);
                    warns_vgd++;
                }

            if (vgb > model->HSMHVvgbMax)
                if (warns_vgb < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vgb|=%g has exceeded Vgb_max=%g\n",
                               vgb, model->HSMHVvgbMax);
                    warns_vgb++;
                }

            if (vds > model->HSMHVvdsMax)
                if (warns_vds < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vds|=%g has exceeded Vds_max=%g\n",
                               vds, model->HSMHVvdsMax);
                    warns_vds++;
                }

            if (vbs > model->HSMHVvbsMax)
                if (warns_vbs < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vbs|=%g has exceeded Vbs_max=%g\n",
                               vbs, model->HSMHVvbsMax);
                    warns_vbs++;
                }

            if (vbd > model->HSMHVvbdMax)
                if (warns_vbd < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "|Vbd|=%g has exceeded Vbd_max=%g\n",
                               vbd, model->HSMHVvbdMax);
                    warns_vbd++;
                }

        }
    }

    return OK;
}

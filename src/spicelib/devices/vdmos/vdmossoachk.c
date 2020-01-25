/**********
Copyright 2013 Dietmar Warning. All rights reserved.
Author:   2013 Dietmar Warning
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vdmosdefs.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/cpdefs.h"


int
VDMOSsoaCheck(CKTcircuit *ckt, GENmodel *inModel)
{
    VDMOSmodel *model = (VDMOSmodel *) inModel;
    VDMOSinstance *here;
    double vgs, vgd, vds;    /* actual mos voltages */
    int maxwarns;
    static int warns_vgs = 0, warns_vgd = 0, warns_vds = 0;

    if (!ckt) {
        warns_vgs = 0;
        warns_vgd = 0;
        warns_vds = 0;
        return OK;
    }

    maxwarns = ckt->CKTsoaMaxWarns;

    for (; model; model = VDMOSnextModel(model)) {

        for (here = VDMOSinstances(model); here; here = VDMOSnextInstance(here)) {

            vgs = ckt->CKTrhsOld [here->VDMOSgNode] -
                  ckt->CKTrhsOld [here->VDMOSsNodePrime];

            vgd = ckt->CKTrhsOld [here->VDMOSgNode] -
                  ckt->CKTrhsOld [here->VDMOSdNodePrime];

            vds = ckt->CKTrhsOld [here->VDMOSdNodePrime] -
                  ckt->CKTrhsOld [here->VDMOSsNodePrime];

            if (!model->VDMOSvgsrMaxGiven) {
                if (fabs(vgs) > model->VDMOSvgsMax)
                    if (warns_vgs < maxwarns) {
                        soa_printf(ckt, (GENinstance*) here,
                                   "Vgs=%g has exceeded Vgs_max=%g\n",
                                   vgs, model->VDMOSvgsMax);
                        warns_vgs++;
                    }
            } else {
                if (model->VDMOStype > 0) {
                    if (vgs > model->VDMOSvgsMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgs_max=%g\n",
                                       vgs, model->VDMOSvgsMax);
                            warns_vgs++;
                        }
                    if (-1*vgs > model->VDMOSvgsrMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgsr_max=%g\n",
                                       vgs, model->VDMOSvgsrMax);
                            warns_vgs++;
                        }
                } else {
                    if (vgs > model->VDMOSvgsrMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgsr_max=%g\n",
                                       vgs, model->VDMOSvgsrMax);
                            warns_vgs++;
                        }
                    if (-1*vgs > model->VDMOSvgsMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgs_max=%g\n",
                                       vgs, model->VDMOSvgsMax);
                            warns_vgs++;
                        }
                }
            }

            if (!model->VDMOSvgdrMaxGiven) {
                if (fabs(vgd) > model->VDMOSvgdMax)
                    if (warns_vgd < maxwarns) {
                        soa_printf(ckt, (GENinstance*) here,
                                   "Vgd=%g has exceeded Vgd_max=%g\n",
                                   vgd, model->VDMOSvgdMax);
                        warns_vgd++;
                    }
            } else {
                if (model->VDMOStype > 0) {
                    if (vgd > model->VDMOSvgdMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgd_max=%g\n",
                                       vgd, model->VDMOSvgdMax);
                            warns_vgd++;
                        }
                    if (-1*vgd > model->VDMOSvgdrMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgdr_max=%g\n",
                                       vgd, model->VDMOSvgdrMax);
                            warns_vgd++;
                        }
                } else {
                    if (vgd > model->VDMOSvgdrMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgdr_max=%g\n",
                                       vgd, model->VDMOSvgdrMax);
                            warns_vgd++;
                        }
                    if (-1*vgd > model->VDMOSvgdMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgd_max=%g\n",
                                       vgd, model->VDMOSvgdMax);
                            warns_vgd++;
                        }
                }
            }

            if (fabs(vds) > model->VDMOSvdsMax)
                if (warns_vds < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "Vds=%g has exceeded Vds_max=%g\n",
                               vds, model->VDMOSvdsMax);
                    warns_vds++;
                }

        }
    }

    return OK;
}

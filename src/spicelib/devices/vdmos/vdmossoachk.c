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
    static int warns_vgs = 0, warns_vgd = 0, warns_vds = 0, warns_id = 0, warns_idr = 0, warns_pd = 0, warns_te = 0;
    double id;  /* current VDMOS current */
    double idr;  /* current reverse diode current */
    double pd;  /* current VDMOS power dissipation */
    double pd_max;  /* maximum VDMOS power */
    double te;  /* current VDMOS temperature */

    if (!ckt) {
        warns_vgs = 0;
        warns_vgd = 0;
        warns_vds = 0;
        warns_id = 0;
        warns_idr = 0;
        warns_pd = 0;
        warns_te = 0;
        return OK;
    }

    maxwarns = ckt->CKTsoaMaxWarns;

    for (; model; model = VDMOSnextModel(model)) {

        for (here = VDMOSinstances(model); here; here = VDMOSnextInstance(here)) {

            vgs = ckt->CKTrhsOld [here->VDMOSgNode] -
                  ckt->CKTrhsOld [here->VDMOSsNode];

            vgd = ckt->CKTrhsOld [here->VDMOSgNode] -
                  ckt->CKTrhsOld [here->VDMOSdNode];

            vds = ckt->CKTrhsOld [here->VDMOSdNode] -
                  ckt->CKTrhsOld [here->VDMOSsNode];

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

            /* max. drain current */
            id = fabs(here->VDMOScd);
            if (model->VDMOSid_maxGiven && id > fabs(model->VDMOSid_max))
                if (warns_id < maxwarns) {
                    soa_printf(ckt, (GENinstance*)here,
                        "Id=%.4g A at Vd=%.4g V has exceeded Id_max=%.4g A\n",
                        id, vds, model->VDMOSid_max);
                    warns_id++;
                }

            /* max. reverse current */
            idr = fabs(*(ckt->CKTstate0 + here->VDIOcurrent) * -1. + here->VDMOScd);
            if (model->VDMOSidr_maxGiven && idr > fabs(model->VDMOSidr_max))
                if (warns_idr < maxwarns) {
                    soa_printf(ckt, (GENinstance*)here,
                        "Idr=%.4g A at Vd=%.4g V has exceeded Idr_max=%.4g A\n",
                        fabs(idr), vds, model->VDMOSidr_max);
                    warns_idr++;
                }

            pd = fabs((id + idr) * vds);
            pd += fabs(*(ckt->CKTstate0 + here->VDMOScqgd) *
                (*(ckt->CKTrhsOld + here->VDMOSgNode) -
                    *(ckt->CKTrhsOld + here->VDMOSdNode)));
            pd += fabs(*(ckt->CKTstate0 + here->VDMOScqgs) *
                (*(ckt->CKTrhsOld + here->VDMOSgNode) -
                    *(ckt->CKTrhsOld + here->VDMOSsNode)));

            /* Calculate max power including derating:
               up to tnom the derating is zero,
               at maximum temp allowed the derating is 100%.
               Device temperature by self-heating or given externally. */
            if (here->VDMOSthermal && model->VDMOSderatingGiven && model->VDMOSpd_maxGiven
                && model->VDMOSte_maxGiven && model->VDMOStnomGiven) {
                te = ckt->CKTrhsOld[here->VDMOStcaseNode];
                if (te < model->VDMOStnom - CONSTCtoK)
                    pd_max = model->VDMOSpd_max;
                else {
                    pd_max = model->VDMOSpd_max - (te - model->VDMOStnom + CONSTCtoK) * model->VDMOSderating;
                    pd_max = (pd_max > 0) ? pd_max : 0.;
                }
                if (pd > pd_max)
                    if (warns_pd < maxwarns) {
                        soa_printf(ckt, (GENinstance*)here,
                            "Pd=%.4g W at Vd=%.4g V and Te=%.4g C has exceeded Pd_max=%.4g W\n",
                            pd, vds, te, pd_max);
                        warns_pd++;
                    }
                if (te > model->VDMOSte_max)
                    if (warns_te < maxwarns) {
                        soa_printf(ckt, (GENinstance*)here,
                            "Te=%.4g C at Vd=%.4g V has exceeded te_max=%.4g C\n",
                            te, vds, model->VDMOSte_max);
                        warns_te++;
                    }

            }
            /* Derating of max allowed power dissipation, without self-heating,
            external temp given by .temp (global) or instance parameter 'temp',
            therefore no temperature limits are calculated */
            else if (!here->VDMOSthermal && model->VDMOSderatingGiven && model->VDMOSpd_maxGiven && model->VDMOStnomGiven) {
                if (here->VDMOStemp < model->VDMOStnom)
                    pd_max = model->VDMOSpd_max;
                else {
                    pd_max = model->VDMOSpd_max - (here->VDMOStemp - model->VDMOStnom) * model->VDMOSderating;
                    pd_max = (pd_max > 0) ? pd_max : 0.;
                }
                if (pd > pd_max)
                    if (warns_pd < maxwarns) {
                        soa_printf(ckt, (GENinstance*)here,
                            "Pd=%.4g W at Vd=%.4g V and Te=%.4g C has exceeded Pd_max=%.4g W\n",
                            pd, vds, here->VDMOStemp - CONSTCtoK, pd_max);
                        warns_pd++;
                    }
            }
            /* No derating, max power is fixed by model parameter pd_max */
            else {
                pd_max = model->VDMOSpd_max;
                if (pd > pd_max)
                    if (warns_pd < maxwarns) {
                        soa_printf(ckt, (GENinstance*)here,
                            "Pd=%.4g W at Vd=%.4g V has exceeded Pd_max=%.4g W\n",
                            pd, vds, pd_max);
                        warns_pd++;
                    }
            }
        }
    }

    return OK;
}

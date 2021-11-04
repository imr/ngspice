/**********
Copyright 2013 Dietmar Warning. All rights reserved.
Author:   2013 Dietmar Warning
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "diodefs.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/cpdefs.h"


int
DIOsoaCheck(CKTcircuit *ckt, GENmodel *inModel)
{
    DIOmodel *model = (DIOmodel *) inModel;
    DIOinstance *here;
    double vd;  /* current diode voltage */
    double id;  /* current diode current */
    double pd;  /* current diode power */
    double pd_max;  /* maximum diode power */
    double te;  /* current diode temperature */
    int maxwarns;
    static int warns_fv = 0, warns_bv = 0, warns_id = 0, warns_pd = 0, warns_te = 0;

    if (!ckt) {
        warns_fv = 0;
        warns_bv = 0;
        warns_id = 0;
        warns_pd = 0;
        warns_te = 0;
        return OK;
    }

    maxwarns = ckt->CKTsoaMaxWarns;

    for (; model; model = DIOnextModel(model)) {

        for (here = DIOinstances(model); here; here = DIOnextInstance(here)) {

            vd = ckt->CKTrhsOld [here->DIOposNode] -
                 ckt->CKTrhsOld [here->DIOnegNode];

            if (vd > model->DIOfv_max)
                if (warns_fv < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "Vd=%.4g V has exceeded Fv_max=%.4g V\n",
                               vd, model->DIOfv_max);
                    warns_fv++;
                }

            if (-vd > model->DIObv_max)
                if (warns_bv < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "Vd=%.4g V has exceeded Bv_max=%.4g V\n",
                               vd, model->DIObv_max);
                    warns_bv++;
                }

            id = fabs(*(ckt->CKTstate0 + here->DIOcurrent));
            if (id > fabs(model->DIOid_max))
                if (warns_id < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "Id=%.4g A at Vd=%.4g V has exceeded Id_max=%.4g A\n",
                               id, vd, model->DIOid_max);
                    warns_id++;
                }


            pd = fabs(*(ckt->CKTstate0 + here->DIOcurrent) *
                      *(ckt->CKTstate0 + here->DIOvoltage) +
                      *(ckt->CKTstate0 + here->DIOcurrent) *
                      *(ckt->CKTstate0 + here->DIOcurrent) / here->DIOtConductance);

            /* Calculate max power including derating:
               up to tnom the derating is zero,
               at maximum temp allowed the derating is 100%.
               Device temperature by self-heating or given externally. */
            if (here->DIOthermal && model->DIOrth0Given && model->DIOpd_maxGiven
                && model->DIOte_maxGiven && model->DIOnomTempGiven) {
                te = ckt->CKTrhsOld[here->DIOtempNode];
                if (te < model->DIOnomTemp)
                    pd_max = model->DIOpd_max;
                else {
                    pd_max = model->DIOpd_max - (te - model->DIOnomTemp) / model->DIOrth0;
                    pd_max = (pd_max > 0) ? pd_max : 0.;
                }
                if (pd > pd_max)
                    if (warns_pd < maxwarns) {
                        soa_printf(ckt, (GENinstance*)here,
                            "Pd=%.4g W at Vd=%.4g V and Te=%.4g C has exceeded Pd_max=%.4g W\n",
                            pd, vd, te, pd_max);
                        warns_pd++;
                    }
                if (te > model->DIOte_max)
                    if (warns_te < maxwarns) {
                        soa_printf(ckt, (GENinstance*)here,
                            "Te=%.4g C at Vd=%.4g V has exceeded te_max=%.4g C\n",
                            te, vd, model->DIOte_max);
                        warns_te++;
                    }

            }
            /* Derating of max allowed power dissipation, without self-heating,
            external temp given by .temp (global) or instance parameter 'temp',
            therefore no temperature limits are calculated */
            else if (!here->DIOthermal && model->DIOrth0Given && model->DIOpd_maxGiven && model->DIOnomTempGiven) {
                if (here->DIOtemp < model->DIOnomTemp)
                    pd_max = model->DIOpd_max;
                else {
                    pd_max = model->DIOpd_max - (here->DIOtemp - model->DIOnomTemp) / model->DIOrth0;
                    pd_max = (pd_max > 0) ? pd_max : 0.;
                }
                if (pd > pd_max)
                    if (warns_pd < maxwarns) {
                        soa_printf(ckt, (GENinstance*)here,
                            "Pd=%.4g W at Vd=%.4g V and Te=%.4g C has exceeded Pd_max=%.4g W\n",
                            pd, vd, here->DIOtemp - CONSTCtoK, pd_max);
                        warns_pd++;
                    }
            }
            /* No derating, max power is fixed by model parameter pd_max */
            else {
                pd_max = model->DIOpd_max;
                if (pd > pd_max)
                    if (warns_pd < maxwarns) {
                        soa_printf(ckt, (GENinstance*)here,
                            "Pd=%.4g W at Vd=%.4g V has exceeded Pd_max=%.4g W\n",
                            pd, vd, pd_max);
                        warns_pd++;
                    }
            }
        }
    }

    return OK;
}

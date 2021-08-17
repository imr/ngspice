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

            vd = ckt->CKTrhsOld [here->DIOposPrimeNode] -
                 ckt->CKTrhsOld [here->DIOnegNode];

            if (vd > model->DIOfv_max)
                if (warns_fv < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "Vj=%g has exceeded Fv_max=%g\n",
                               vd, model->DIOfv_max);
                    warns_fv++;
                }

            if (-vd > model->DIObv_max)
                if (warns_bv < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "Vj=%g has exceeded Bv_max=%g\n",
                               vd, model->DIObv_max);
                    warns_bv++;
                }

            id = fabs(*(ckt->CKTstate0 + here->DIOcurrent));
            if (id > fabs(model->DIOid_max))
                if (warns_id < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "Id=%g has exceeded Id_max=%g\n",
                               id, model->DIOid_max);
                    warns_id++;
                }


            pd = fabs(*(ckt->CKTstate0 + here->DIOcurrent) *
                      *(ckt->CKTstate0 + here->DIOvoltage) +
                      *(ckt->CKTstate0 + here->DIOcurrent) *
                      *(ckt->CKTstate0 + here->DIOcurrent) / here->DIOtConductance);
            if (pd > fabs(model->DIOpd_max))
                if (warns_pd < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "Pd=%g has exceeded Pd_max=%g\n",
                               pd, model->DIOpd_max);
                    warns_pd++;
                }

            te = here->DIOtemp - CONSTCtoK;
            if (te > model->DIOte_max)
                if (warns_te < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "Te=%g has exceeded te_max=%g\n",
                               te, model->DIOte_max);
                    warns_te++;
                }

        }

    }

    return OK;
}

/**********
Copyright 2013 Dietmar Warning. All rights reserved.
Author:   2013 Dietmar Warning
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vbicdefs.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/cpdefs.h"


int
VBICsoaCheck(CKTcircuit *ckt, GENmodel *inModel)
{
    VBICmodel *model = (VBICmodel *) inModel;
    VBICinstance *here;
    double vbe, vbc, vce, vsub;    /* actual bjt voltages */
    int maxwarns;
    static int warns_vbe = 0, warns_vbc = 0, warns_vce = 0, warns_vsub = 0, warns_op = 0;

    if (!ckt) {
        warns_vbe = 0;
        warns_vbc = 0;
        warns_vce = 0;
        warns_vsub = 0;
        warns_op = 0;
        return OK;
    }

    maxwarns = ckt->CKTsoaMaxWarns;

    for (; model; model = VBICnextModel(model)) {

        for (here = VBICinstances(model); here; here = VBICnextInstance(here)) {

            vbe = fabs(ckt->CKTrhsOld[here->VBICbaseNode] -
                ckt->CKTrhsOld[here->VBICemitNode]);
            vbc = fabs(ckt->CKTrhsOld[here->VBICbaseNode] -
                ckt->CKTrhsOld[here->VBICcollNode]);
            vce = fabs(ckt->CKTrhsOld[here->VBICcollNode] -
                ckt->CKTrhsOld[here->VBICemitNode]);
            vsub = fabs(ckt->CKTrhsOld[here->VBICcollNode] -
                ckt->CKTrhsOld[here->VBICsubsNode]);

            if (vbe > model->VBICvbeMax)
                if (warns_vbe < maxwarns) {
                    soa_printf(ckt, (GENinstance*)here,
                        "|Vbe|=%g has exceeded Vbe_max=%g\n",
                        vbe, model->VBICvbeMax);
                    warns_vbe++;
                }

            if (vbc > model->VBICvbcMax)
                if (warns_vbc < maxwarns) {
                    soa_printf(ckt, (GENinstance*)here,
                        "|Vbc|=%g has exceeded Vbc_max=%g\n",
                        vbc, model->VBICvbcMax);
                    warns_vbc++;
                }

            if (vce > model->VBICvceMax)
                if (warns_vce < maxwarns) {
                    soa_printf(ckt, (GENinstance*)here,
                        "|Vce|=%g has exceeded Vce_max=%g\n",
                        vce, model->VBICvceMax);
                    warns_vce++;
                }

            if (vsub > model->VBICvsubMax)
                if (warns_vsub < maxwarns) {
                    soa_printf(ckt, (GENinstance*)here,
                        "|Vcs|=%g has exceeded Vcs_max=%g\n",
                        vsub, model->VBICvsubMax);
                    warns_vsub++;
                }

            /* substrate diode is forward biased */
            if (model->VBICtype * (ckt->CKTrhsOld[here->VBICsubsNode] -
                ckt->CKTrhsOld[here->VBICcollNode]) > model->VBICvsubfwdMax) {
                /* substrate leakage */
                if (warns_vsub < maxwarns) {
                    soa_printf(ckt, (GENinstance*)here,
                        "substrate juntion is forward biased\n");
                    warns_vsub++;
                }
            }

            /* operating point information */
            if (ckt->CKTsoaCheck == 2) {
                if (vbe <= model->VBICvbefwdMax && vbc <= model->VBICvbefwdMax) {
                    /*off*/
                    if (warns_op < maxwarns) {
                        soa_printf(ckt, (GENinstance*)here,
                            "device is off\n");
                        warns_op++;
                    }
                }
                else if (vbe > model->VBICvbefwdMax && vbc > model->VBICvbefwdMax) {
                    /*saturation*/
                    if (warns_op < maxwarns) {
                        soa_printf(ckt, (GENinstance*)here,
                            "device is in saturation\n");
                        warns_op++;
                    }
                }
                else if (vbe > model->VBICvbefwdMax && vbc <= model->VBICvbefwdMax) {
                    /*forward*/
                    if (warns_op < maxwarns) {
                        soa_printf(ckt, (GENinstance*)here,
                            "device is forward biased\n");
                        warns_op++;
                    }
                }
                else if (vbe <= model->VBICvbefwdMax && vbc > model->VBICvbefwdMax) {
                    /*reverse*/
                    if (warns_op < maxwarns) {
                        soa_printf(ckt, (GENinstance*)here,
                            "device is reverse biased\n");
                        warns_op++;
                    }
                }
            }
        }
    }

    return OK;
}

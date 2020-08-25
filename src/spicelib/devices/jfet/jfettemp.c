/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Sydney University mods Copyright(c) 1989 Anthony E. Parker, David J. Skellern
        Laboratory for Communication Science Engineering
        Sydney University Department of Electrical Engineering, Australia
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "jfetdefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
JFETtemp(GENmodel *inModel, CKTcircuit *ckt)
        /* Pre-process the model parameters after a possible change
         */
{
    JFETmodel *model = (JFETmodel*)inModel;
    JFETinstance *here;
    double xfc;
    double vt;
    double vtnom;
    double kt,kt1;
    double arg,arg1;
    double fact1,fact2;
    double egfet,egfet1;
    double pbfact,pbfact1;
    double gmanew,gmaold;
    double ratio1;
    double pbo;
    double cjfact,cjfact1;

    /*  loop through all the diode models */
    for( ; model != NULL; model = JFETnextModel(model)) {

        if(!(model->JFETtnomGiven)) {
            model->JFETtnom = ckt->CKTnomTemp;
        }
        vtnom = CONSTKoverQ * model->JFETtnom;
        fact1 = model->JFETtnom/REFTEMP;
        kt1 = CONSTboltz * model->JFETtnom;
        egfet1 = 1.16-(7.02e-4*model->JFETtnom*model->JFETtnom)/
                (model->JFETtnom+1108);
        arg1 = -egfet1/(kt1+kt1)+1.1150877/(CONSTboltz*(REFTEMP+REFTEMP));
        pbfact1 = -2*vtnom * (1.5*log(fact1)+CHARGE*arg1);
        pbo = (model->JFETgatePotential-pbfact1)/fact1;
        gmaold = (model->JFETgatePotential-pbo)/pbo;
        cjfact = 1/(1+.5*(4e-4*(model->JFETtnom-REFTEMP)-gmaold));

        if(model->JFETdrainResist != 0) {
            model->JFETdrainConduct = 1/model->JFETdrainResist;
        } else {
            model->JFETdrainConduct = 0;
        }
        if(model->JFETsourceResist != 0) {
            model->JFETsourceConduct = 1/model->JFETsourceResist;
        } else {
            model->JFETsourceConduct = 0;
        }
        if(model->JFETdepletionCapCoeff >.95) {
            SPfrontEnd->IFerrorf (ERR_WARNING,
                    "%s: Depletion cap. coefficient too large, limited to .95",
                    model->JFETmodName);
            model->JFETdepletionCapCoeff = .95;
        }

        xfc = log(1 - model->JFETdepletionCapCoeff);
        model->JFETf2 = exp((1+.5)*xfc);
        model->JFETf3 = 1 - model->JFETdepletionCapCoeff * (1 + .5);
        /* Modification for Sydney University JFET model */
        model->JFETbFac = (1 - model->JFETb)
                / (model->JFETgatePotential - model->JFETthreshold);
        /* end Sydney University mod */

        /* loop through all the instances of the model */
        for (here = JFETinstances(model); here != NULL ;
                here=JFETnextInstance(here)) {

            if(!(here->JFETdtempGiven)) {
                here->JFETdtemp = 0.0;
            }
            if(!(here->JFETtempGiven)) {
                here->JFETtemp = ckt->CKTtemp + here->JFETdtemp;
            }
            vt = here->JFETtemp * CONSTKoverQ;
            fact2 = here->JFETtemp/REFTEMP;
            ratio1 = here->JFETtemp/model->JFETtnom -1;
            if (model->JFETxtiGiven) {
                here->JFETtSatCur = model->JFETgateSatCurrent * exp(ratio1*model->JFETeg/vt) * pow(ratio1+1,model->JFETxti);
            } else {
                here->JFETtSatCur = model->JFETgateSatCurrent * exp(ratio1*model->JFETeg/vt);
            }
            here->JFETtCGS = model->JFETcapGS * cjfact;
            here->JFETtCGD = model->JFETcapGD * cjfact;
            kt = CONSTboltz*here->JFETtemp;
            egfet = 1.16-(7.02e-4*here->JFETtemp*here->JFETtemp)/
                    (here->JFETtemp+1108);
            arg = -egfet/(kt+kt) + 1.1150877/(CONSTboltz*(REFTEMP+REFTEMP));
            pbfact = -2 * vt * (1.5*log(fact2)+CHARGE*arg);
            here->JFETtGatePot = fact2 * pbo + pbfact;
            gmanew = (here->JFETtGatePot-pbo)/pbo;
            cjfact1 = 1+.5*(4e-4*(here->JFETtemp-REFTEMP)-gmanew);
            here->JFETtCGS *= cjfact1;
            here->JFETtCGD *= cjfact1;

            here->JFETcorDepCap = model->JFETdepletionCapCoeff *
                    here->JFETtGatePot;
            here->JFETf1 = here->JFETtGatePot * (1 - exp((1-.5)*xfc))/(1-.5);
            here->JFETvcrit = vt * log(vt/(CONSTroot2 * here->JFETtSatCur));

            if (model->JFETvtotcGiven) {
                here->JFETtThreshold = model->JFETthreshold + model->JFETvtotc*(here->JFETtemp-model->JFETtnom);
            } else {
                here->JFETtThreshold = model->JFETthreshold - model->JFETtcv*(here->JFETtemp-model->JFETtnom);
            }
            if (model->JFETbetatceGiven) {
                here->JFETtBeta = model->JFETbeta * pow(1.01,model->JFETbetatce*(here->JFETtemp-model->JFETtnom));
            } else {
                here->JFETtBeta = model->JFETbeta * pow(here->JFETtemp/model->JFETtnom,model->JFETbex);
            }
        }
    }
    return(OK);
}

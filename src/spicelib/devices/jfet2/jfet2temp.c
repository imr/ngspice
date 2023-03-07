/**********
Base on jfettemp.c
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

Modified to add PS model and new parameter definitions ( Anthony E. Parker ).
See https://ngspice.sourceforge.io/external-documents/models/psfet.pdf
   Copyright 1994  Macquarie University, Sydney Australia.
   10 Feb 1994: Call to PSinstanceinit() added
                Change gatePotential to phi and used rs and rd for 
                sourceResist and drainResist, and fc for depletionCapCoef
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "jfet2defs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "psmodel.h"
#include "ngspice/suffix.h"

int
JFET2temp(GENmodel *inModel, CKTcircuit *ckt)
        /* Pre-process the model parameters after a possible change
         */
{
    JFET2model *model = (JFET2model*)inModel;
    JFET2instance *here;
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
    for( ; model != NULL; model = JFET2nextModel(model)) {

        if(!(model->JFET2tnomGiven)) {
            model->JFET2tnom = ckt->CKTnomTemp;
        }
        vtnom = CONSTKoverQ * model->JFET2tnom;
        fact1 = model->JFET2tnom/REFTEMP;
        kt1 = CONSTboltz * model->JFET2tnom;
        egfet1 = 1.16-(7.02e-4*model->JFET2tnom*model->JFET2tnom)/
                (model->JFET2tnom+1108);
        arg1 = -egfet1/(kt1+kt1)+1.1150877/(CONSTboltz*(REFTEMP+REFTEMP));
        pbfact1 = -2*vtnom * (1.5*log(fact1)+CHARGE*arg1);
        pbo = (model->JFET2phi-pbfact1)/fact1;
        gmaold = (model->JFET2phi-pbo)/pbo;
        cjfact = 1/(1+.5*(4e-4*(model->JFET2tnom-REFTEMP)-gmaold));

        if(model->JFET2rd != 0) {
            model->JFET2drainConduct = 1/model->JFET2rd;
        } else {
            model->JFET2drainConduct = 0;
        }
        if(model->JFET2rs != 0) {
            model->JFET2sourceConduct = 1/model->JFET2rs;
        } else {
            model->JFET2sourceConduct = 0;
        }
        if(model->JFET2fc >.95) {
            SPfrontEnd->IFerrorf (ERR_WARNING,
                    "%s: Depletion cap. coefficient too large, limited to .95",
                    model->JFET2modName);
            model->JFET2fc = .95;
        }

        xfc = log(1 - model->JFET2fc);
        model->JFET2f2 = exp((1+.5)*xfc);
        model->JFET2f3 = 1 - model->JFET2fc * (1 + .5);

        /* loop through all the instances of the model */
        for (here = JFET2instances(model); here != NULL ;
                here=JFET2nextInstance(here)) {

            if(!(here->JFET2dtempGiven)) {
                here->JFET2dtemp = 0.0;
            }

            if(!(here->JFET2tempGiven)) {
                here->JFET2temp = ckt->CKTtemp + here->JFET2dtemp;
            }

            vt = here->JFET2temp * CONSTKoverQ;
            fact2 = here->JFET2temp/REFTEMP;
            ratio1 = here->JFET2temp/model->JFET2tnom -1;
            here->JFET2tSatCur = model->JFET2is * exp(ratio1*1.11/vt);
            here->JFET2tCGS = model->JFET2capgs * cjfact;
            here->JFET2tCGD = model->JFET2capgd * cjfact;
            kt = CONSTboltz*here->JFET2temp;
            egfet = 1.16-(7.02e-4*here->JFET2temp*here->JFET2temp)/
                    (here->JFET2temp+1108);
            arg = -egfet/(kt+kt) + 1.1150877/(CONSTboltz*(REFTEMP+REFTEMP));
            pbfact = -2 * vt * (1.5*log(fact2)+CHARGE*arg);
            here->JFET2tGatePot = fact2 * pbo + pbfact;
            gmanew = (here->JFET2tGatePot-pbo)/pbo;
            cjfact1 = 1+.5*(4e-4*(here->JFET2temp-REFTEMP)-gmanew);
            here->JFET2tCGS *= cjfact1;
            here->JFET2tCGD *= cjfact1;

            here->JFET2corDepCap = model->JFET2fc * here->JFET2tGatePot;
            here->JFET2f1 = here->JFET2tGatePot * (1 - exp((1-.5)*xfc))/(1-.5);
            here->JFET2vcrit = vt * log(vt/(CONSTroot2 * here->JFET2tSatCur));

            PSinstanceinit(model, here);
            
        }
    }
    return(OK);
}

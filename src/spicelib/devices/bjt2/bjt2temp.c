/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "smpdefs.h"
#include "bjt2defs.h"
#include "const.h"
#include "sperror.h"
#include "ifsim.h"
#include "suffix.h"


/* ARGSUSED */
int
BJT2temp(GENmodel *inModel, CKTcircuit *ckt)
        /* 
      * Pre-compute many useful parameters
      */

{
    BJT2model *model = (BJT2model *)inModel;
    BJT2instance *here;
    double xfc;
    double vt;
    double ratlog;
    double ratio1;
    double factlog;
    double bfactor;
    double factor;
    double fact1,fact2;
    double pbo,pbfact;
    double gmaold,gmanew;
    double egfet;
    double arg;
    double dtemp;

    /*  loop through all the bipolar models */
    for( ; model != NULL; model = model->BJT2nextModel ) {

        if(!model->BJT2tnomGiven) model->BJT2tnom = ckt->CKTnomTemp;
        fact1 = model->BJT2tnom/REFTEMP;

        if(!model->BJT2leakBEcurrentGiven) {
            if(model->BJT2c2Given) {
                model->BJT2leakBEcurrent = model->BJT2c2 * model->BJT2satCur;
            } else {
                model->BJT2leakBEcurrent = 0;
            }
        }
        if(!model->BJT2leakBCcurrentGiven) {
            if(model->BJT2c4Given) {
                model->BJT2leakBCcurrent = model->BJT2c4 * model->BJT2satCur;
            } else {
                model->BJT2leakBCcurrent = 0;
            }
        }
        if(!model->BJT2minBaseResistGiven) {
            model->BJT2minBaseResist = model->BJT2baseResist;
        }

/*
 * COMPATABILITY WARNING!
 * special note:  for backward compatability to much older models, spice 2G
 * implemented a special case which checked if B-E leakage saturation
 * current was >1, then it was instead a the B-E leakage saturation current
 * divided by IS, and multiplied it by IS at this point.  This was not
 * handled correctly in the 2G code, and there is some question on its 
 * reasonability, since it is also undocumented, so it has been left out
 * here.  It could easily be added with 1 line.  (The same applies to the B-C
 * leakage saturation current).   TQ  6/29/84
 */
            
        if(model->BJT2earlyVoltFGiven && model->BJT2earlyVoltF != 0) {
            model->BJT2invEarlyVoltF = 1/model->BJT2earlyVoltF;
        } else {
            model->BJT2invEarlyVoltF = 0;
        }
        if(model->BJT2rollOffFGiven && model->BJT2rollOffF != 0) {
            model->BJT2invRollOffF = 1/model->BJT2rollOffF;
        } else {
            model->BJT2invRollOffF = 0;
        }
        if(model->BJT2earlyVoltRGiven && model->BJT2earlyVoltR != 0) {
            model->BJT2invEarlyVoltR = 1/model->BJT2earlyVoltR;
        } else {
            model->BJT2invEarlyVoltR = 0;
        }
        if(model->BJT2rollOffRGiven && model->BJT2rollOffR != 0) {
            model->BJT2invRollOffR = 1/model->BJT2rollOffR;
        } else {
            model->BJT2invRollOffR = 0;
        }
        if(model->BJT2collectorResistGiven && model->BJT2collectorResist != 0) {
            model->BJT2collectorConduct = 1/model->BJT2collectorResist;
        } else {
            model->BJT2collectorConduct = 0;
        }
        if(model->BJT2emitterResistGiven && model->BJT2emitterResist != 0) {
            model->BJT2emitterConduct = 1/model->BJT2emitterResist;
        } else {
            model->BJT2emitterConduct = 0;
        }
        if(model->BJT2transitTimeFVBCGiven && model->BJT2transitTimeFVBC != 0) {
            model->BJT2transitTimeVBCFactor =1/ (model->BJT2transitTimeFVBC*1.44);
        } else {
            model->BJT2transitTimeVBCFactor = 0;
        }
        model->BJT2excessPhaseFactor = (model->BJT2excessPhase/
            (180.0/M_PI)) * model->BJT2transitTimeF;
        if(model->BJT2depletionCapCoeffGiven) {
            if(model->BJT2depletionCapCoeff>.9999)  {
                model->BJT2depletionCapCoeff=.9999;
                (*(SPfrontEnd->IFerror))(ERR_WARNING,
                        "BJT2 model %s, parameter fc limited to 0.9999",
                        &(model->BJT2modName));
            }
        } else {
            model->BJT2depletionCapCoeff=.5;
        }
        xfc = log(1-model->BJT2depletionCapCoeff);
        model->BJT2f2 = exp((1 + model->BJT2junctionExpBE) * xfc);
        model->BJT2f3 = 1 - model->BJT2depletionCapCoeff * 
                (1 + model->BJT2junctionExpBE);
        model->BJT2f6 = exp((1+model->BJT2junctionExpBC)*xfc);
        model->BJT2f7 = 1 - model->BJT2depletionCapCoeff * 
                (1 + model->BJT2junctionExpBC);

        /* loop through all the instances of the model */
        for (here = model->BJT2instances; here != NULL ;
                here=here->BJT2nextInstance) {
            if (here->BJT2owner != ARCHme) continue;
		
	    if(!here->BJT2dtempGiven) 
	       here->BJT2dtemp = 0.0;
	       	
            if(!here->BJT2tempGiven) 
	       here->BJT2temp = ckt->CKTtemp + here->BJT2dtemp;
            
	    vt = here->BJT2temp * CONSTKoverQ;
            fact2 = here->BJT2temp/REFTEMP;
            egfet = 1.16-(7.02e-4*here->BJT2temp*here->BJT2temp)/
                    (here->BJT2temp+1108);
            arg = -egfet/(2*CONSTboltz*here->BJT2temp)+
                    1.1150877/(CONSTboltz*(REFTEMP+REFTEMP));
            pbfact = -2*vt*(1.5*log(fact2)+CHARGE*arg);

            ratlog = log(here->BJT2temp/model->BJT2tnom);
            ratio1 = here->BJT2temp/model->BJT2tnom -1;
            factlog = ratio1 * model->BJT2energyGap/vt + 
                    model->BJT2tempExpIS*ratlog;
            factor = exp(factlog);
            here->BJT2tSatCur = model->BJT2satCur * factor;
            here->BJT2tSubSatCur = model->BJT2subSatCur * factor;
            bfactor = exp(ratlog*model->BJT2betaExp);
            here->BJT2tBetaF = model->BJT2betaF * bfactor;
            here->BJT2tBetaR = model->BJT2betaR * bfactor;
            here->BJT2tBEleakCur = model->BJT2leakBEcurrent * 
                    exp(factlog/model->BJT2leakBEemissionCoeff)/bfactor;
            here->BJT2tBCleakCur = model->BJT2leakBCcurrent * 
                    exp(factlog/model->BJT2leakBCemissionCoeff)/bfactor;

            dtemp = here->BJT2temp - model->BJT2tnom;
            if(model->BJT2emitterResistGiven && model->BJT2emitterResist != 0) {
              factor = 1.0 + (model->BJT2reTempCoeff1)*dtemp + 
                      (model->BJT2reTempCoeff2)*dtemp*dtemp;
              here -> BJT2tEmitterConduct = 1/(model->BJT2emitterResist * factor);
            } else {
              here -> BJT2tEmitterConduct = 0;
            }
            if(model->BJT2collectorResistGiven && model->BJT2collectorResist != 0) {
              factor = 1.0 + (model->BJT2rcTempCoeff1)*dtemp + 
                      (model->BJT2rcTempCoeff2)*dtemp*dtemp;
              here -> BJT2tCollectorConduct = 1/(model->BJT2collectorResist * factor);
            } else {
              here -> BJT2tCollectorConduct = 0;
            }
            factor = 1.0 + (model->BJT2rbTempCoeff1)*dtemp + 
                    (model->BJT2rbTempCoeff2)*dtemp*dtemp;
            here -> BJT2tBaseResist = model->BJT2baseResist * factor;
            factor = 1.0 + (model->BJT2rbmTempCoeff1)*dtemp + 
                    (model->BJT2rbmTempCoeff2)*dtemp*dtemp;
            here -> BJT2tMinBaseResist = model->BJT2minBaseResist * factor;

            pbo = (model->BJT2potentialBE-pbfact)/fact1;
            gmaold = (model->BJT2potentialBE-pbo)/pbo;
            here->BJT2tBEcap = model->BJT2depletionCapBE/
                    (1+model->BJT2junctionExpBE*
                    (4e-4*(model->BJT2tnom-REFTEMP)-gmaold));
            here->BJT2tBEpot = fact2 * pbo+pbfact;
            gmanew = (here->BJT2tBEpot-pbo)/pbo;
            here->BJT2tBEcap *= 1+model->BJT2junctionExpBE*
                    (4e-4*(here->BJT2temp-REFTEMP)-gmanew);

            pbo = (model->BJT2potentialBC-pbfact)/fact1;
            gmaold = (model->BJT2potentialBC-pbo)/pbo;
            here->BJT2tBCcap = model->BJT2depletionCapBC/
                    (1+model->BJT2junctionExpBC*
                    (4e-4*(model->BJT2tnom-REFTEMP)-gmaold));
            here->BJT2tBCpot = fact2 * pbo+pbfact;
            gmanew = (here->BJT2tBCpot-pbo)/pbo;
            here->BJT2tBCcap *= 1+model->BJT2junctionExpBC*
                    (4e-4*(here->BJT2temp-REFTEMP)-gmanew);
            pbo = (model->BJT2potentialSubstrate-pbfact)/fact1;
            gmaold = (model->BJT2potentialSubstrate-pbo)/pbo;
            here->BJT2tSubcap = model->BJT2capSub/
                    (1+model->BJT2exponentialSubstrate*
                    (4e-4*(model->BJT2tnom-REFTEMP)-gmaold));
            here->BJT2tSubpot = fact2 * pbo+pbfact;
            gmanew = (here->BJT2tSubpot-pbo)/pbo;
            here->BJT2tSubcap *= 1+model->BJT2exponentialSubstrate*
                    (4e-4*(here->BJT2temp-REFTEMP)-gmanew);
            here->BJT2tDepCap = model->BJT2depletionCapCoeff * here->BJT2tBEpot;
            here->BJT2tf1 = here->BJT2tBEpot * (1 - exp((1 - 
                    model->BJT2junctionExpBE) * xfc)) / 
                    (1 - model->BJT2junctionExpBE);
            here->BJT2tf4 = model->BJT2depletionCapCoeff * here->BJT2tBCpot;
            here->BJT2tf5 = here->BJT2tBCpot * (1 - exp((1 -
                    model->BJT2junctionExpBC) * xfc)) /
                    (1 - model->BJT2junctionExpBC);
            here->BJT2tVcrit = vt *
                     log(vt / (CONSTroot2*here->BJT2tSatCur*here->BJT2area));
            here->BJT2tSubVcrit = vt *
                     log(vt / (CONSTroot2*here->BJT2tSubSatCur*here->BJT2area));
        }
    }
    return(OK);
}

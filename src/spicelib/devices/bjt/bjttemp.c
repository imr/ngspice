/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "smpdefs.h"
#include "bjtdefs.h"
#include "const.h"
#include "sperror.h"
#include "ifsim.h"
#include "suffix.h"


/* ARGSUSED */
int
BJTtemp(GENmodel *inModel, CKTcircuit *ckt)
        /* Pre-compute many useful parameters
      */

{
    BJTmodel *model = (BJTmodel *)inModel;
    BJTinstance *here;
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

    /*  loop through all the bipolar models */
    for( ; model != NULL; model = model->BJTnextModel ) {

        if(!model->BJTtnomGiven) model->BJTtnom = ckt->CKTnomTemp;
        fact1 = model->BJTtnom/REFTEMP;

        if(!model->BJTleakBEcurrentGiven) {
            if(model->BJTc2Given) {
                model->BJTleakBEcurrent = model->BJTc2 * model->BJTsatCur;
            } else {
                model->BJTleakBEcurrent = 0;
            }
        }
        if(!model->BJTleakBCcurrentGiven) {
            if(model->BJTc4Given) {
                model->BJTleakBCcurrent = model->BJTc4 * model->BJTsatCur;
            } else {
                model->BJTleakBCcurrent = 0;
            }
        }
        if(!model->BJTminBaseResistGiven) {
            model->BJTminBaseResist = model->BJTbaseResist;
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
            
        if(model->BJTearlyVoltFGiven && model->BJTearlyVoltF != 0) {
            model->BJTinvEarlyVoltF = 1/model->BJTearlyVoltF;
        } else {
            model->BJTinvEarlyVoltF = 0;
        }
        if(model->BJTrollOffFGiven && model->BJTrollOffF != 0) {
            model->BJTinvRollOffF = 1/model->BJTrollOffF;
        } else {
            model->BJTinvRollOffF = 0;
        }
        if(model->BJTearlyVoltRGiven && model->BJTearlyVoltR != 0) {
            model->BJTinvEarlyVoltR = 1/model->BJTearlyVoltR;
        } else {
            model->BJTinvEarlyVoltR = 0;
        }
        if(model->BJTrollOffRGiven && model->BJTrollOffR != 0) {
            model->BJTinvRollOffR = 1/model->BJTrollOffR;
        } else {
            model->BJTinvRollOffR = 0;
        }
        if(model->BJTcollectorResistGiven && model->BJTcollectorResist != 0) {
            model->BJTcollectorConduct = 1/model->BJTcollectorResist;
        } else {
            model->BJTcollectorConduct = 0;
        }
        if(model->BJTemitterResistGiven && model->BJTemitterResist != 0) {
            model->BJTemitterConduct = 1/model->BJTemitterResist;
        } else {
            model->BJTemitterConduct = 0;
        }
        if(model->BJTtransitTimeFVBCGiven && model->BJTtransitTimeFVBC != 0) {
            model->BJTtransitTimeVBCFactor =1/ (model->BJTtransitTimeFVBC*1.44);
        } else {
            model->BJTtransitTimeVBCFactor = 0;
        }
        model->BJTexcessPhaseFactor = (model->BJTexcessPhase/
            (180.0/M_PI)) * model->BJTtransitTimeF;
        if(model->BJTdepletionCapCoeffGiven) {
            if(model->BJTdepletionCapCoeff>.9999)  {
                model->BJTdepletionCapCoeff=.9999;
                (*(SPfrontEnd->IFerror))(ERR_WARNING,
                        "BJT model %s, parameter fc limited to 0.9999",
                        &(model->BJTmodName));
            }
        } else {
            model->BJTdepletionCapCoeff=.5;
        }
        xfc = log(1-model->BJTdepletionCapCoeff);
        model->BJTf2 = exp((1 + model->BJTjunctionExpBE) * xfc);
        model->BJTf3 = 1 - model->BJTdepletionCapCoeff * 
                (1 + model->BJTjunctionExpBE);
        model->BJTf6 = exp((1+model->BJTjunctionExpBC)*xfc);
        model->BJTf7 = 1 - model->BJTdepletionCapCoeff * 
                (1 + model->BJTjunctionExpBC);

        /* loop through all the instances of the model */
        for (here = model->BJTinstances; here != NULL ;
                here=here->BJTnextInstance) {
	    if (here->BJTowner != ARCHme) continue;

            if(!here->BJTdtempGiven) 
	       here->BJTdtemp = 0.0;
	       
            if(!here->BJTtempGiven) 
	       here->BJTtemp = ckt->CKTtemp + here->BJTdtemp;
            
	    vt = here->BJTtemp * CONSTKoverQ;
            fact2 = here->BJTtemp/REFTEMP;
            egfet = 1.16-(7.02e-4*here->BJTtemp*here->BJTtemp)/
                    (here->BJTtemp+1108);
            arg = -egfet/(2*CONSTboltz*here->BJTtemp)+
                    1.1150877/(CONSTboltz*(REFTEMP+REFTEMP));
            pbfact = -2*vt*(1.5*log(fact2)+CHARGE*arg);

            ratlog = log(here->BJTtemp/model->BJTtnom);
            ratio1 = here->BJTtemp/model->BJTtnom -1;
            factlog = ratio1 * model->BJTenergyGap/vt + 
                    model->BJTtempExpIS*ratlog;
            factor = exp(factlog);
            here->BJTtSatCur = model->BJTsatCur * factor;
            bfactor = exp(ratlog*model->BJTbetaExp);
            here->BJTtBetaF = model->BJTbetaF * bfactor;
            here->BJTtBetaR = model->BJTbetaR * bfactor;
            here->BJTtBEleakCur = model->BJTleakBEcurrent * 
                    exp(factlog/model->BJTleakBEemissionCoeff)/bfactor;
            here->BJTtBCleakCur = model->BJTleakBCcurrent * 
                    exp(factlog/model->BJTleakBCemissionCoeff)/bfactor;

            pbo = (model->BJTpotentialBE-pbfact)/fact1;
            gmaold = (model->BJTpotentialBE-pbo)/pbo;
            here->BJTtBEcap = model->BJTdepletionCapBE/
                    (1+model->BJTjunctionExpBE*
                    (4e-4*(model->BJTtnom-REFTEMP)-gmaold));
            here->BJTtBEpot = fact2 * pbo+pbfact;
            gmanew = (here->BJTtBEpot-pbo)/pbo;
            here->BJTtBEcap *= 1+model->BJTjunctionExpBE*
                    (4e-4*(here->BJTtemp-REFTEMP)-gmanew);

            pbo = (model->BJTpotentialBC-pbfact)/fact1;
            gmaold = (model->BJTpotentialBC-pbo)/pbo;
            here->BJTtBCcap = model->BJTdepletionCapBC/
                    (1+model->BJTjunctionExpBC*
                    (4e-4*(model->BJTtnom-REFTEMP)-gmaold));
            here->BJTtBCpot = fact2 * pbo+pbfact;
            gmanew = (here->BJTtBCpot-pbo)/pbo;
            here->BJTtBCcap *= 1+model->BJTjunctionExpBC*
                    (4e-4*(here->BJTtemp-REFTEMP)-gmanew);

            here->BJTtDepCap = model->BJTdepletionCapCoeff * here->BJTtBEpot;
            here->BJTtf1 = here->BJTtBEpot * (1 - exp((1 - 
                    model->BJTjunctionExpBE) * xfc)) / 
                    (1 - model->BJTjunctionExpBE);
            here->BJTtf4 = model->BJTdepletionCapCoeff * here->BJTtBCpot;
            here->BJTtf5 = here->BJTtBCpot * (1 - exp((1 -
                    model->BJTjunctionExpBC) * xfc)) /
                    (1 - model->BJTjunctionExpBC);
            here->BJTtVcrit = vt *
                     log(vt / (CONSTroot2*here->BJTtSatCur*here->BJTarea));
            
        }
    }
    return(OK);
}

/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/smpdefs.h"
#include "bjtdefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/ifsim.h"
#include "ngspice/suffix.h"


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
    double bfactor=1.0;
    double factor;
    double fact1,fact2;
    double pbo,pbfact;
    double gmaold,gmanew;
    double egfet;
    double arg;
    double dt;

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

        if(model->BJTtransitTimeFVBCGiven && model->BJTtransitTimeFVBC != 0) {
            model->BJTtransitTimeVBCFactor =1/(model->BJTtransitTimeFVBC*1.44);
        } else {
            model->BJTtransitTimeVBCFactor = 0;
        }
        model->BJTexcessPhaseFactor = (model->BJTexcessPhase/
            (180.0/M_PI)) * model->BJTtransitTimeF;
        if(model->BJTdepletionCapCoeffGiven) {
            if(model->BJTdepletionCapCoeff>.9999)  {
                model->BJTdepletionCapCoeff=.9999;
                SPfrontEnd->IFerror (ERR_WARNING,
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

            dt = here->BJTtemp - model->BJTtnom;

            if(model->BJTearlyVoltFGiven && model->BJTearlyVoltF != 0) {
                here->BJTtinvEarlyVoltF = 1/(model->BJTearlyVoltF * (1+model->BJTtvaf1*dt+model->BJTtvaf2*dt*dt));
            } else {
                here->BJTtinvEarlyVoltF = 0;
            }
            if(model->BJTrollOffFGiven && model->BJTrollOffF != 0) {
                here->BJTtinvRollOffF = 1/(model->BJTrollOffF * (1+model->BJTtikf1*dt+model->BJTtikf2*dt*dt));
            } else {
                here->BJTtinvRollOffF = 0;
            }
            if(model->BJTearlyVoltRGiven && model->BJTearlyVoltR != 0) {
                here->BJTtinvEarlyVoltR = 1/(model->BJTearlyVoltR * (1+model->BJTtvar1*dt+model->BJTtvar2*dt*dt));
            } else {
                here->BJTtinvEarlyVoltR = 0;
            }
            if(model->BJTrollOffRGiven && model->BJTrollOffR != 0) {
                here->BJTtinvRollOffR = 1/(model->BJTrollOffR * (1+model->BJTtikr1*dt+model->BJTtikr2*dt*dt));
            } else {
                here->BJTtinvRollOffR = 0;
            }
            if(model->BJTcollectorResistGiven && model->BJTcollectorResist != 0) {
                here->BJTtcollectorConduct = 1/(model->BJTcollectorResist * (1+model->BJTtrc1*dt+model->BJTtrc2*dt*dt));
            } else {
                here->BJTtcollectorConduct = 0;
            }
            if(model->BJTemitterResistGiven && model->BJTemitterResist != 0) {
                here->BJTtemitterConduct = 1/(model->BJTemitterResist * (1+model->BJTtre1*dt+model->BJTtre2*dt*dt));
            } else {
                here->BJTtemitterConduct = 0;
            }

            here->BJTtbaseResist = model->BJTbaseResist * (1+model->BJTtrb1*dt+model->BJTtrb2*dt*dt);
            here->BJTtminBaseResist = model->BJTminBaseResist*(1+model->BJTtrm1*dt+model->BJTtrm2*dt*dt);
            here->BJTtbaseCurrentHalfResist = model->BJTbaseCurrentHalfResist * (1+model->BJTtirb1*dt+model->BJTtirb2*dt*dt);
            here->BJTtemissionCoeffF = model->BJTemissionCoeffF * (1+model->BJTtnf1*dt+model->BJTtnf2*dt*dt);
            here->BJTtemissionCoeffR = model->BJTemissionCoeffR * (1+model->BJTtnr1*dt+model->BJTtnr2*dt*dt);
            here->BJTtleakBEemissionCoeff = model->BJTleakBEemissionCoeff * (1+model->BJTtne1*dt+model->BJTtne2*dt*dt);
            here->BJTtleakBCemissionCoeff = model->BJTleakBCemissionCoeff * (1+model->BJTtnc1*dt+model->BJTtnc2*dt*dt);
            here->BJTttransitTimeHighCurrentF = model->BJTtransitTimeHighCurrentF * (1+model->BJTtitf1*dt+model->BJTtitf2*dt*dt);
            here->BJTttransitTimeF = model->BJTtransitTimeF * (1+model->BJTttf1*dt+model->BJTttf2*dt*dt);
            here->BJTttransitTimeR = model->BJTtransitTimeR * (1+model->BJTttr1*dt+model->BJTttr2*dt*dt);
            here->BJTtjunctionExpBE = model->BJTjunctionExpBE * (1+model->BJTtmje1*dt+model->BJTtmje2*dt*dt);
            here->BJTtjunctionExpBC = model->BJTjunctionExpBC * (1+model->BJTtmjc1*dt+model->BJTtmjc2*dt*dt);
            here->BJTtjunctionExpSub = model->BJTexponentialSubstrate * (1+model->BJTtmjs1*dt+model->BJTtmjs2*dt*dt);
            here->BJTtemissionCoeffS = model->BJTemissionCoeffS * (1+model->BJTtns1*dt+model->BJTtns2*dt*dt);

            if ((model->BJTtlev == 0) || (model->BJTtlev == 1)) {
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
              here->BJTtSubSatCur = model->BJTsubSatCur * factor;
            } else if (model->BJTtlev == 3) {
              here->BJTtSatCur = pow(model->BJTsatCur,(1+model->BJTtis1*dt+model->BJTtis2*dt*dt));
            }

            if (model->BJTtlev == 0) {
                bfactor = exp(ratlog*model->BJTbetaExp);
            } else if (model->BJTtlev == 1) {
                bfactor = 1+model->BJTbetaExp*dt;
            }
            if ((model->BJTtbf1Given)||(model->BJTtbf2Given))
                here->BJTtBetaF = model->BJTbetaF * (1+model->BJTtbf1*dt+model->BJTtbf2*dt*dt);
            else
                here->BJTtBetaF = model->BJTbetaF * bfactor;
            if ((model->BJTtbr1Given)||(model->BJTtbr2Given))
                here->BJTtBetaR = model->BJTbetaR * (1+model->BJTtbr1*dt+model->BJTtbr2*dt*dt);
            else
                here->BJTtBetaR = model->BJTbetaR * bfactor;

            if ((model->BJTtlev == 0) || (model->BJTtlev == 1)) {
              here->BJTtBEleakCur = model->BJTleakBEcurrent *
                  exp(factlog/model->BJTleakBEemissionCoeff)/bfactor;
              here->BJTtBCleakCur = model->BJTleakBCcurrent *
                  exp(factlog/model->BJTleakBCemissionCoeff)/bfactor;
            } else if (model->BJTtlev == 3) {
              here->BJTtBEleakCur = pow(model->BJTleakBEcurrent,(1+model->BJTtise1*dt+model->BJTtise2*dt*dt));
              here->BJTtBCleakCur = pow(model->BJTleakBCcurrent,(1+model->BJTtisc1*dt+model->BJTtisc2*dt*dt));
            }

            if (model->BJTtlevc == 0) {
                pbo = (model->BJTpotentialBE-pbfact)/fact1;
                gmaold = (model->BJTpotentialBE-pbo)/pbo;
                here->BJTtBEcap = model->BJTdepletionCapBE/
                    (1+here->BJTtjunctionExpBE*
                    (4e-4*(model->BJTtnom-REFTEMP)-gmaold));
                here->BJTtBEpot = fact2 * pbo+pbfact;
                gmanew = (here->BJTtBEpot-pbo)/pbo;
                here->BJTtBEcap *= 1+here->BJTtjunctionExpBE*
                    (4e-4*(here->BJTtemp-REFTEMP)-gmanew);
            } else if (model->BJTtlevc == 1) {
                here->BJTtBEcap = model->BJTdepletionCapBE*
                    (1+model->BJTcte*dt);			
                here->BJTtBEpot = model->BJTpotentialBE - model->BJTtvje*dt;
            }
            if (model->BJTtlevc == 0) {
                pbo = (model->BJTpotentialBC-pbfact)/fact1;
                gmaold = (model->BJTpotentialBC-pbo)/pbo;
                here->BJTtBCcap = model->BJTdepletionCapBC/
                    (1+here->BJTtjunctionExpBC*
                    (4e-4*(model->BJTtnom-REFTEMP)-gmaold));
                here->BJTtBCpot = fact2 * pbo+pbfact;
                gmanew = (here->BJTtBCpot-pbo)/pbo;
                here->BJTtBCcap *= 1+here->BJTtjunctionExpBC*
                    (4e-4*(here->BJTtemp-REFTEMP)-gmanew);
            } else if (model->BJTtlevc == 1) {
                here->BJTtBCcap = model->BJTdepletionCapBC*
                    (1+model->BJTctc*dt);			
                here->BJTtBCpot = model->BJTpotentialBC - model->BJTtvjc*dt;
            }
            if (model->BJTtlevc == 0) {
                pbo = (model->BJTpotentialSubstrate-pbfact)/fact1;
                gmaold = (model->BJTpotentialSubstrate-pbo)/pbo;
                here->BJTtSubcap = model->BJTcapSub/
                        (1+here->BJTtjunctionExpSub*
                        (4e-4*(model->BJTtnom-REFTEMP)-gmaold));
                here->BJTtSubpot = fact2 * pbo+pbfact;
                gmanew = (here->BJTtSubpot-pbo)/pbo;
                here->BJTtSubcap *= 1+here->BJTtjunctionExpSub*
                    (4e-4*(here->BJTtemp-REFTEMP)-gmanew);
            } else if (model->BJTtlevc == 1) {
                here->BJTtSubcap = model->BJTcapSub*
                    (1+model->BJTcts*dt);			
                here->BJTtSubpot = model->BJTpotentialSubstrate - model->BJTtvjs*dt;
            }

            here->BJTtDepCap = model->BJTdepletionCapCoeff * here->BJTtBEpot;
            here->BJTtf1 = here->BJTtBEpot * (1 - exp((1 -
                    here->BJTtjunctionExpBE) * xfc)) /
                    (1 - here->BJTtjunctionExpBE);
            here->BJTtf4 = model->BJTdepletionCapCoeff * here->BJTtBCpot;
            here->BJTtf5 = here->BJTtBCpot * (1 - exp((1 -
                    here->BJTtjunctionExpBC) * xfc)) /
                    (1 - here->BJTtjunctionExpBC);
            here->BJTtVcrit = vt *
                     log(vt / (CONSTroot2*here->BJTtSatCur*here->BJTarea));
            here->BJTtSubVcrit = vt *
                     log(vt / (CONSTroot2*here->BJTtSubSatCur*here->BJTarea));

        }
    }
    return(OK);
}

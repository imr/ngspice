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
    double vtnom;
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
    for( ; model != NULL; model = BJTnextModel(model)) {

        if(!model->BJTtnomGiven) model->BJTtnom = ckt->CKTnomTemp;
        vtnom = CONSTKoverQ * model->BJTtnom;
        fact1 = model->BJTtnom/REFTEMP;

        if(!model->BJTminBaseResistGiven) {
            model->BJTminBaseResist = model->BJTbaseResist;
        }

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
                SPfrontEnd->IFerrorf (ERR_WARNING,
                        "BJT model %s, parameter fc limited to 0.9999",
                        model->BJTmodName);
            }
        } else {
            model->BJTdepletionCapCoeff=.5;
        }
        xfc = log(1-model->BJTdepletionCapCoeff);

        /* loop through all the instances of the model */
        for (here = BJTinstances(model); here != NULL ;
                here=BJTnextInstance(here)) {

            double arg1, pbfact1, egfet1;
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
                here->BJTtinvRollOffF /= here->BJTarea;
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
                here->BJTtinvRollOffR /= here->BJTarea;
            } else {
                here->BJTtinvRollOffR = 0;
            }
            if(model->BJTcollectorResistGiven && model->BJTcollectorResist != 0) {
                here->BJTtcollectorConduct = 1/(model->BJTcollectorResist * (1+model->BJTtrc1*dt+model->BJTtrc2*dt*dt));
                here->BJTtcollectorConduct *= here->BJTarea;
            } else {
                here->BJTtcollectorConduct = 0;
            }
            if(model->BJTemitterResistGiven && model->BJTemitterResist != 0) {
                here->BJTtemitterConduct = 1/(model->BJTemitterResist * (1+model->BJTtre1*dt+model->BJTtre2*dt*dt));
                here->BJTtemitterConduct *= here->BJTarea;
            } else {
                here->BJTtemitterConduct = 0;
            }

            here->BJTtbaseResist = model->BJTbaseResist * (1+model->BJTtrb1*dt+model->BJTtrb2*dt*dt);
            here->BJTtbaseResist /= here->BJTarea;
            here->BJTtminBaseResist = model->BJTminBaseResist*(1+model->BJTtrm1*dt+model->BJTtrm2*dt*dt);
            here->BJTtminBaseResist /= here->BJTarea;
            here->BJTtbaseCurrentHalfResist = model->BJTbaseCurrentHalfResist * (1+model->BJTtirb1*dt+model->BJTtirb2*dt*dt);
            here->BJTtbaseCurrentHalfResist *= here->BJTarea;
            here->BJTtemissionCoeffF = model->BJTemissionCoeffF * (1+model->BJTtnf1*dt+model->BJTtnf2*dt*dt);
            here->BJTtemissionCoeffR = model->BJTemissionCoeffR * (1+model->BJTtnr1*dt+model->BJTtnr2*dt*dt);
            here->BJTtleakBEemissionCoeff = model->BJTleakBEemissionCoeff * (1+model->BJTtne1*dt+model->BJTtne2*dt*dt);
            here->BJTtleakBCemissionCoeff = model->BJTleakBCemissionCoeff * (1+model->BJTtnc1*dt+model->BJTtnc2*dt*dt);
            here->BJTttransitTimeHighCurrentF = model->BJTtransitTimeHighCurrentF * (1+model->BJTtitf1*dt+model->BJTtitf2*dt*dt);
            here->BJTttransitTimeHighCurrentF *= here->BJTarea;
            here->BJTttransitTimeF = model->BJTtransitTimeF * (1+model->BJTttf1*dt+model->BJTttf2*dt*dt);
            here->BJTttransitTimeR = model->BJTtransitTimeR * (1+model->BJTttr1*dt+model->BJTttr2*dt*dt);
            here->BJTtjunctionExpBE = model->BJTjunctionExpBE * (1+model->BJTtmje1*dt+model->BJTtmje2*dt*dt);
            if (here->BJTtjunctionExpBE > 0.999) {/* limit required due to line 310 */
                here->BJTtjunctionExpBE = 0.999;
                fprintf(stderr, "Warning: parameter mje (including tempco) of model %s is limited to 0.999\n", model->gen.GENmodName);
            }
            here->BJTtjunctionExpBC = model->BJTjunctionExpBC * (1+model->BJTtmjc1*dt+model->BJTtmjc2*dt*dt);
            if (here->BJTtjunctionExpBC > 0.999) {/* limit required due to line 314 */
                here->BJTtjunctionExpBC = 0.999;
                fprintf(stderr, "Warning: parameter mjc (including tempco) of model %s is limited to 0.999\n", model->gen.GENmodName);
            }
            here->BJTtjunctionExpSub = model->BJTexponentialSubstrate * (1+model->BJTtmjs1*dt+model->BJTtmjs2*dt*dt);
            if (here->BJTtjunctionExpSub > 0.999) {/* limit required due to line 732 in bjtload.c */
                here->BJTtjunctionExpSub = 0.999;
                fprintf(stderr, "Warning: parameter mjs (including tempco) of model %s is limited to 0.999\n", model->gen.GENmodName);
            }
            here->BJTtemissionCoeffS = model->BJTemissionCoeffS * (1+model->BJTtns1*dt+model->BJTtns2*dt*dt);

            vt = here->BJTtemp * CONSTKoverQ;
            fact2 = here->BJTtemp/REFTEMP;
            egfet = 1.16-(7.02e-4*here->BJTtemp*here->BJTtemp)/
                    (here->BJTtemp+1108);
            arg = -egfet/(2*CONSTboltz*here->BJTtemp)+
                    1.1150877/(CONSTboltz*(REFTEMP+REFTEMP));
            pbfact = -2*vt*(1.5*log(fact2)+CHARGE*arg);
            egfet1 = 1.16-(7.02e-4*model->BJTtnom*model->BJTtnom)/
                    (model->BJTtnom+1108);
            arg1 = -egfet1/(2*CONSTboltz*model->BJTtnom)+
                    1.1150877/(CONSTboltz*(REFTEMP+REFTEMP));
            pbfact1 = -2*vtnom*(1.5*log(fact1)+CHARGE*arg1);

            ratlog = log(here->BJTtemp/model->BJTtnom);
            ratio1 = here->BJTtemp/model->BJTtnom -1;
            factlog = ratio1 * model->BJTenergyGap/vt +
                    model->BJTtempExpIS*ratlog;
            if ((model->BJTtlev == 0) || (model->BJTtlev == 1)) {
                factor = exp(factlog);
                here->BJTtSatCur = here->BJTarea * model->BJTsatCur * factor;
                if ((model->BJTBEsatCurGiven) && (model->BJTBCsatCurGiven)) {
                    factor = exp(factlog / model->BJTemissionCoeffF);
                    here->BJTBEtSatCur = here->BJTarea * model->BJTBEsatCur * factor;
                } else {
                    here->BJTBEtSatCur = here->BJTtSatCur;
                }
                if ((model->BJTBEsatCurGiven) && (model->BJTBCsatCurGiven)) {
                    factor = exp(factlog / model->BJTemissionCoeffR);
                    here->BJTBCtSatCur = model->BJTBCsatCur * factor;
                } else {
                    here->BJTBCtSatCur = here->BJTtSatCur;
                }
                if (model->BJTsubSatCurGiven)
                    here->BJTtSubSatCur = model->BJTsubSatCur * factor;
            } else if (model->BJTtlev == 3) {
                here->BJTtSatCur = here->BJTarea * pow(model->BJTsatCur,(1+model->BJTtis1*dt+model->BJTtis2*dt*dt));
                if ((model->BJTBEsatCurGiven) && (model->BJTBCsatCurGiven)) {
                    here->BJTBEtSatCur = here->BJTarea * pow(model->BJTBEsatCur,(1+model->BJTtis1*dt+model->BJTtis2*dt*dt));
                } else {
                    here->BJTBEtSatCur = here->BJTtSatCur;
                }
                if ((model->BJTBEsatCurGiven) && (model->BJTBCsatCurGiven)) {
                    here->BJTBCtSatCur = pow(model->BJTBCsatCur,(1+model->BJTtis1*dt+model->BJTtis2*dt*dt));
                } else {
                    here->BJTBCtSatCur = here->BJTtSatCur;
                }
                if (model->BJTsubSatCurGiven)
                    here->BJTtSubSatCur = pow(model->BJTsubSatCur,(1+model->BJTtiss1*dt+model->BJTtiss2*dt*dt));
            }
            if (model->BJTsubs == VERTICAL) {
                here->BJTBCtSatCur *= here->BJTareab;
            } else {
                here->BJTBCtSatCur *= here->BJTareac;
            }
            if (model->BJTsubSatCurGiven) {
                if ((model->BJTBEsatCurGiven) && (model->BJTBCsatCurGiven)) {
                    if (model->BJTsubs == VERTICAL) {
                        here->BJTtSubSatCur *= here->BJTareac;
                    } else {
                        here->BJTtSubSatCur *= here->BJTareab;
                    }
                } else {
                    here->BJTtSubSatCur *= here->BJTarea;
                }
            }

            if (model->BJTintCollResistGiven) {
                if (model->BJTquasimod == 1) {
                    double rT=here->BJTtemp/model->BJTtnom;
                    here->BJTtintCollResist=model->BJTintCollResist*pow(rT,model->BJTtempExpRCI);
                    here->BJTtepiSatVoltage=model->BJTepiSatVoltage*pow(rT,model->BJTtempExpVO);
                    double xvar1=pow(rT,model->BJTtempExpIS);
                    double xvar2=-model->BJTenergyGapQS*(1.0-rT)/vt;
                    double xvar3=exp(xvar2);
                    here->BJTtepiDoping=model->BJTepiDoping*xvar1*xvar3;
                } else {
                    here->BJTtintCollResist=model->BJTintCollResist;
                    here->BJTtepiSatVoltage=model->BJTepiSatVoltage;
                    here->BJTtepiDoping=model->BJTepiDoping;
                }
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
              here->BJTtBEleakCur = here->BJTarea * model->BJTleakBEcurrent *
                  exp(factlog/model->BJTleakBEemissionCoeff)/bfactor;
              here->BJTtBCleakCur = model->BJTleakBCcurrent *
                  exp(factlog/model->BJTleakBCemissionCoeff)/bfactor;
            } else if (model->BJTtlev == 3) {
              here->BJTtBEleakCur = here->BJTarea * pow(model->BJTleakBEcurrent,(1+model->BJTtise1*dt+model->BJTtise2*dt*dt));
              here->BJTtBCleakCur = pow(model->BJTleakBCcurrent,(1+model->BJTtisc1*dt+model->BJTtisc2*dt*dt));
            }
            if (model->BJTsubs == VERTICAL) {
                here->BJTtBCleakCur *= here->BJTareab;
            } else {
                here->BJTtBCleakCur *= here->BJTareac;
            }

            if (model->BJTtlevc == 0) {
                pbo = (model->BJTpotentialBE-pbfact1)/fact1;
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
            here->BJTtBEcap *= here->BJTarea;
            if (model->BJTtlevc == 0) {
                pbo = (model->BJTpotentialBC-pbfact1)/fact1;
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
            if (model->BJTsubs == VERTICAL)
                here->BJTtBCcap *= here->BJTareab;
            else
                here->BJTtBCcap *= here->BJTareac;
            if (model->BJTtlevc == 0) {
                pbo = (model->BJTpotentialSubstrate-pbfact1)/fact1;
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
            if (model->BJTsubs == VERTICAL)
                here->BJTtSubcap *= here->BJTareac;
            else
                here->BJTtSubcap *= here->BJTareab;

            here->BJTtDepCap = model->BJTdepletionCapCoeff * here->BJTtBEpot;
            here->BJTtf1 = here->BJTtBEpot * (1 - exp((1 -
                    here->BJTtjunctionExpBE) * xfc)) /
                    (1 - here->BJTtjunctionExpBE);
            here->BJTtf4 = model->BJTdepletionCapCoeff * here->BJTtBCpot;
            here->BJTtf5 = here->BJTtBCpot * (1 - exp((1 -
                    here->BJTtjunctionExpBC) * xfc)) /
                    (1 - here->BJTtjunctionExpBC);
            here->BJTtVcrit = vt *
                     log(vt / (CONSTroot2*here->BJTtSatCur));
            if (model->BJTsubSatCurGiven)
                here->BJTtSubVcrit = vt *
                         log(vt / (CONSTroot2*here->BJTtSubSatCur));
            here->BJTtf2 = exp((1 + here->BJTtjunctionExpBE) * xfc);
            here->BJTtf3 = 1 - model->BJTdepletionCapCoeff *
                    (1 + here->BJTtjunctionExpBE);
            here->BJTtf6 = exp((1 + here->BJTtjunctionExpBC)*xfc);
            here->BJTtf7 = 1 - model->BJTdepletionCapCoeff *
                    (1 + here->BJTtjunctionExpBC);

        }
    }
    return(OK);
}

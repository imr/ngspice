/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

    /* perform the temperature update to the diode */

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "diodefs.h"
#include "const.h"
#include "sperror.h"
#include "suffix.h"

int
DIOtemp(inModel,ckt)
    GENmodel *inModel;
    register CKTcircuit *ckt;
{
    register DIOmodel *model = (DIOmodel*)inModel;
    double xfc;
    double vte;
    double cbv;
    double xbv;
    double xcbv;
    double tol;
    double vt;
    double vtnom;
    register DIOinstance *here;
    register int iter;
    char *emsg;

    /*  loop through all the diode models */
    for( ; model != NULL; model = model->DIOnextModel ) {
        if(!model->DIOnomTempGiven) {
            model->DIOnomTemp = ckt->CKTnomTemp;
        }
        vtnom = CONSTKoverQ * model->DIOnomTemp;
        /* limit grading coeff to max of .9 */
        if(model->DIOgradingCoeff>.9) {
            (*(SPfrontEnd->IFerror))(ERR_WARNING,
                    "%s: grading coefficient too large, limited to 0.9",
                    &(model->DIOmodName));
            model->DIOgradingCoeff=.9;
        }
        /* limit activation energy to min of .1 */
        if(model->DIOactivationEnergy<.1) {
            (*(SPfrontEnd->IFerror))(ERR_WARNING,
                    "%s: activation energy too small, limited to 0.1",
                    &(model->DIOmodName));
            model->DIOactivationEnergy=.1;
        }
        /* limit depletion cap coeff to max of .95 */
        if(model->DIOdepletionCapCoeff>.95) {
            (*(SPfrontEnd->IFerror))(ERR_WARNING,
                    "%s: coefficient Fc too large, limited to 0.95",
                    &(model->DIOmodName));
            model->DIOdepletionCapCoeff=.95;
        }
        if(!model->DIOresistGiven || model->DIOresist==0) {
            model->DIOconductance = 0;
        } else {
            model->DIOconductance = 1/model->DIOresist;
        }
        xfc=log(1-model->DIOdepletionCapCoeff);
        for(here=model->DIOinstances;here;here=here->DIOnextInstance) {
            double egfet1,arg1,fact1,pbfact1,pbo,gmaold;
            double fact2,pbfact,arg,egfet,gmanew;
	    if (here->DIOowner != ARCHme) continue;

            /* loop through all the instances */
            if(!here->DIOtempGiven) here->DIOtemp = ckt->CKTtemp;
            vt = CONSTKoverQ * here->DIOtemp;
            /* this part gets really ugly - I won't even try to
             * explain these equations */
            fact2 = here->DIOtemp/REFTEMP;
            egfet = 1.16-(7.02e-4*here->DIOtemp*here->DIOtemp)/
                    (here->DIOtemp+1108);
            arg = -egfet/(2*CONSTboltz*here->DIOtemp) +
                    1.1150877/(CONSTboltz*(REFTEMP+REFTEMP));
            pbfact = -2*vt*(1.5*log(fact2)+CHARGE*arg);
            egfet1 = 1.16 - (7.02e-4*model->DIOnomTemp*model->DIOnomTemp)/
                    (model->DIOnomTemp+1108);
            arg1 = -egfet1/(CONSTboltz*2*model->DIOnomTemp) + 
                    1.1150877/(2*CONSTboltz*REFTEMP);
            fact1 = model->DIOnomTemp/REFTEMP;
            pbfact1 = -2 * vtnom*(1.5*log(fact1)+CHARGE*arg1);
            pbo = (model->DIOjunctionPot-pbfact1)/fact1;
            gmaold = (model->DIOjunctionPot -pbo)/pbo;
            here->DIOtJctCap = model->DIOjunctionCap/
                    (1+model->DIOgradingCoeff*
                    (400e-6*(model->DIOnomTemp-REFTEMP)-gmaold) );
            here->DIOtJctPot = pbfact+fact2*pbo;
            gmanew = (here->DIOtJctPot-pbo)/pbo;
            here->DIOtJctCap *= 1+model->DIOgradingCoeff*
                    (400e-6*(here->DIOtemp-REFTEMP)-gmanew);

            here->DIOtSatCur = model->DIOsatCur *  exp( 
                    ((here->DIOtemp/model->DIOnomTemp)-1) *
                    model->DIOactivationEnergy/(model->DIOemissionCoeff*vt) +
                    model->DIOsaturationCurrentExp/model->DIOemissionCoeff*
                    log(here->DIOtemp/model->DIOnomTemp) );
            /* the defintion of f1, just recompute after temperature adjusting
             * all the variables used in it */
            here->DIOtF1=here->DIOtJctPot*
                    (1-exp((1-model->DIOgradingCoeff)*xfc))/
                    (1-model->DIOgradingCoeff);
            /* same for Depletion Capacitance */
            here->DIOtDepCap=model->DIOdepletionCapCoeff*
                    here->DIOtJctPot;
            /* and Vcrit */
            vte=model->DIOemissionCoeff*vt;
            here->DIOtVcrit=vte*log(vte/(CONSTroot2*here->DIOtSatCur));
            /* and now to copute the breakdown voltage, again, using
             * temperature adjusted basic parameters */
            if (model->DIObreakdownVoltageGiven){
                cbv=model->DIObreakdownCurrent;
                if (cbv < here->DIOtSatCur*model->DIObreakdownVoltage/vt){
                    cbv=here->DIOtSatCur*model->DIObreakdownVoltage/vt;
                    emsg = MALLOC(100);
                    if(emsg == (char *)NULL) return(E_NOMEM);
                    (void)sprintf(emsg,
            "%%s: breakdown current increased to %g to resolve incompatability",
                            cbv);
                    (*(SPfrontEnd->IFerror))(ERR_WARNING,emsg,&(here->DIOname));
                    FREE(emsg);
                    (*(SPfrontEnd->IFerror))(ERR_WARNING,
                            "with specified saturation current",(IFuid*)NULL);
                    xbv=model->DIObreakdownVoltage;
                } else {
                    tol=ckt->CKTreltol*cbv;
                    xbv=model->DIObreakdownVoltage-vt*log(1+cbv/
                            here->DIOtSatCur);
                    iter=0;
                    for(iter=0 ; iter < 25 ; iter++) {
                        xbv=model->DIObreakdownVoltage-vt*log(cbv/
                                here->DIOtSatCur+1-xbv/vt);
                        xcbv=here->DIOtSatCur*(exp((model->DIObreakdownVoltage
                                -xbv)/vt)-1+xbv/vt);
                        if (fabs(xcbv-cbv) <= tol) goto matched;
                    }
                    emsg = MALLOC(100);
                    if(emsg == (char *)NULL) return(E_NOMEM);
                    (void)sprintf(emsg,
"%%s: unable to match forward and reverse diode regions: bv = %g, ibv = %g",
                            xbv,xcbv);
                    (*(SPfrontEnd->IFerror))(ERR_WARNING,emsg,&here->DIOname);
                    FREE(emsg);
                }
                matched:
                here->DIOtBrkdwnV = xbv;
            }
        }
        model->DIOf2=exp((1+model->DIOgradingCoeff)*xfc);
        model->DIOf3=1-model->DIOdepletionCapCoeff*
                (1+model->DIOgradingCoeff);
    }
    return(OK);
}

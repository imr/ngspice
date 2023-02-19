/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
Modified by Paolo Nenzi 2003 and Dietmar Warning 2012
**********/

/* load the diode structure with those pointers needed later
 * for fast matrix loading
 */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "diodefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/fteext.h"

int
DIOsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
{
    DIOmodel *model = (DIOmodel*)inModel;
    DIOinstance *here;
    int error;
    CKTnode *tmp;
    double scale;

    if (!cp_getvar("scale", CP_REAL, &scale, 0))
        scale = 1;

    /*  loop through all the diode models */
    for( ; model != NULL; model = DIOnextModel(model)) {

        if(!model->DIOlevelGiven) {
            model->DIOlevel = 1;
        }
        if(!model->DIOemissionCoeffGiven) {
            model->DIOemissionCoeff = 1;
        }
        if(!model->DIOsatCurGiven) {
            model->DIOsatCur = 1e-14;
        }
        if(!model->DIOsatSWCurGiven) {
            model->DIOsatSWCur = 0.0;
        }
        if(!model->DIOswEmissionCoeffGiven) {
            model->DIOswEmissionCoeff = 1;
        }
        if(!model->DIObreakdownCurrentGiven) {
            model->DIObreakdownCurrent = 1e-3;
        }
        if(!model->DIOjunctionPotGiven){
            model->DIOjunctionPot = 1;
        }
        if(!model->DIOgradingCoeffGiven) {
            model->DIOgradingCoeff = .5;
        }
        if(!model->DIOgradCoeffTemp1Given) {
            model->DIOgradCoeffTemp1 = 0.0;
        }
        if(!model->DIOgradCoeffTemp2Given) {
            model->DIOgradCoeffTemp2 = 0.0;
        }
        if(!model->DIOdepletionCapCoeffGiven) {
            model->DIOdepletionCapCoeff = .5;
        }
        if(!model->DIOdepletionSWcapCoeffGiven) {
            model->DIOdepletionSWcapCoeff = .5;
        }
        if(!model->DIOtransitTimeGiven) {
            model->DIOtransitTime = 0;
        }
        if(!model->DIOtranTimeTemp1Given) {
            model->DIOtranTimeTemp1 = 0.0;
        }
        if(!model->DIOtranTimeTemp2Given) {
            model->DIOtranTimeTemp2 = 0.0;
        }
        if(!model->DIOjunctionCapGiven) {
            model->DIOjunctionCap = 0;
        }
        if(!model->DIOjunctionSWCapGiven) {
            model->DIOjunctionSWCap = 0;
        }
        if(!model->DIOjunctionSWPotGiven){
            model->DIOjunctionSWPot = 1;
        }
        if(!model->DIOgradingSWCoeffGiven) {
            model->DIOgradingSWCoeff = .33;
        }
        if(model->DIOforwardKneeCurrentGiven) {
            if (model->DIOforwardKneeCurrent < ckt->CKTepsmin) {
                model->DIOforwardKneeCurrentGiven = FALSE;
                printf("Warning: IKF too small - model effect disabled!\n");
            }
        }
        if(model->DIOreverseKneeCurrentGiven) {
            if (model->DIOreverseKneeCurrent < ckt->CKTepsmin) {
                model->DIOreverseKneeCurrentGiven = FALSE;
                printf("Warning: IKK too small - model effect disabled!\n");
            }
        }
        if(!model->DIObrkdEmissionCoeffGiven) {
            model->DIObrkdEmissionCoeff = model->DIOemissionCoeff;
        }
        if(!model->DIOtlevGiven) {
            model->DIOtlev = 0;
        }
        if(!model->DIOtlevcGiven) {
            model->DIOtlevc = 0;
        }
        if(!model->DIOactivationEnergyGiven) {
            model->DIOactivationEnergy = 1.11;
        }
        if(!model->DIOsaturationCurrentExpGiven) {
            model->DIOsaturationCurrentExp = 3;
        }
        if(!model->DIOctaGiven) {
            model->DIOcta = 0.0;
        }
        if(!model->DIOctpGiven) {
            model->DIOctp = 0.0;
        }
        if(!model->DIOtpbGiven) {
            model->DIOtpb = 0.0;
        }
        if(!model->DIOtphpGiven) {
            model->DIOtphp = 0.0;
        }
        if(!model->DIOfNcoefGiven) {
            model->DIOfNcoef = 0.0;
        }
        if(!model->DIOfNexpGiven) {
            model->DIOfNexp = 1.0;
        }
        if(!model->DIOresistTemp1Given) {
            model->DIOresistTemp1 = 0.0;
        }
        if(!model->DIOresistTemp2Given) {
            model->DIOresistTemp2 = 0.0;
        }
        if(!model->DIOtcvGiven) {
            model->DIOtcv = 0.0;
        }
        if(!model->DIOareaGiven) {
            model->DIOarea = 1.0;
        }
        if(!model->DIOpjGiven) {
            model->DIOpj = 0.0;
        }
        if(!model->DIOtunSatCurGiven) {
            model->DIOtunSatCur = 0.0;
        }
        if(!model->DIOtunSatSWCurGiven) {
            model->DIOtunSatSWCur = 0.0;
        }
        if(!model->DIOtunEmissionCoeffGiven) {
            model->DIOtunEmissionCoeff = 30.0;
        }
        if(!model->DIOtunSaturationCurrentExpGiven) {
            model->DIOtunSaturationCurrentExp = 3.0;
        }
        if(!model->DIOtunEGcorrectionFactorGiven) {
            model->DIOtunEGcorrectionFactor = 1.0;
        }
        if(!model->DIOfv_maxGiven) {
            model->DIOfv_max = 1e99;
        }
        if(!model->DIObv_maxGiven) {
            model->DIObv_max = 1e99;
        }
        if(!model->DIOid_maxGiven) {
            model->DIOid_max = 1e99;
        }
        if(!model->DIOpd_maxGiven) {
            model->DIOpd_max = 1e99;
        }
        if(!model->DIOte_maxGiven) {
            model->DIOte_max = 1e99;
        }
        if(!model->DIOrecEmissionCoeffGiven) {
            model->DIOrecEmissionCoeff = 2;
        }
        if(!model->DIOrecSatCurGiven) {
            model->DIOrecSatCur = 1e-14;
        }

        /* set lower limit of saturation current */
        if (model->DIOsatCur < ckt->CKTepsmin)
            model->DIOsatCur = ckt->CKTepsmin;

        if(!model->DIOnomTempGiven) {
            model->DIOnomTemp = ckt->CKTnomTemp;
        }

        if((!model->DIOresistGiven) || (model->DIOresist==0)) {
            model->DIOconductance = 0.0;
        } else {
            model->DIOconductance = 1/model->DIOresist;
        }

        if (!model->DIOrth0Given) {
            model->DIOrth0 = 0;
        }
        if (!model->DIOcth0Given) {
            model->DIOcth0 = 1e-5;
        }

        if(!model->DIOlengthMetalGiven) {
            model->DIOlengthMetal = 0.0;
        }
        if(!model->DIOlengthPolyGiven) {
            model->DIOlengthPoly = 0.0;
        }
        if(!model->DIOwidthMetalGiven) {
            model->DIOwidthMetal = 0.0;
        }
        if(!model->DIOwidthPolyGiven) {
            model->DIOwidthPoly = 0.0;
        }
        if(!model->DIOmetalOxideThickGiven) {
            model->DIOmetalOxideThick = 1e-06; /* m */
        }
        if(!model->DIOpolyOxideThickGiven) {
            model->DIOpolyOxideThick = 1e-06; /* m */
        }
        if(!model->DIOmetalMaskOffsetGiven) {
            model->DIOmetalMaskOffset = 0.0;
        }
        if(!model->DIOpolyMaskOffsetGiven) {
            model->DIOpolyMaskOffset = 0.0;
        }

        /* loop through all the instances of the model */
        for (here = DIOinstances(model); here != NULL ;
                here=DIOnextInstance(here)) {

            if(!here->DIOareaGiven) {
                if((!here->DIOwGiven) && (!here->DIOlGiven))  {
                    here->DIOarea = model->DIOarea;
                } else {
                    here->DIOarea = 1;
                }
            }
            if(!here->DIOpjGiven) {
                if((!here->DIOwGiven) && (!here->DIOlGiven))  {
                    here->DIOpj = model->DIOpj;
                } else {
                    here->DIOpj = 0;
                }
            }
            if(!here->DIOmGiven) {
                here->DIOm = 1;
            }

            here->DIOarea = here->DIOarea * here->DIOm;
            here->DIOpj = here->DIOpj * here->DIOm;
            here->DIOcmetal = 0.0;
            here->DIOcpoly = 0.0;
            if (model->DIOlevel == 3) {
                double wm, lm, wp, lp;
                if((here->DIOwGiven) && (here->DIOlGiven))  {
                    here->DIOarea = here->DIOw * here->DIOl * here->DIOm;
                    here->DIOpj = (2 * here->DIOw + 2 * here->DIOl) * here->DIOm;
                }
                here->DIOarea = here->DIOarea * scale * scale;
                here->DIOpj = here->DIOpj * scale;
                if (here->DIOwidthMetalGiven)
                    wm = here->DIOwidthMetal;
                else
                    wm = model->DIOwidthMetal;
                if (here->DIOlengthMetalGiven)
                    lm = here->DIOlengthMetal;
                else
                    lm = model->DIOlengthMetal;
                if (here->DIOwidthPolyGiven)
                    wp = here->DIOwidthPoly;
                else
                    wp = model->DIOwidthPoly;
                if (here->DIOlengthPolyGiven)
                    lp = here->DIOlengthPoly;
                else
                    lp = model->DIOlengthPoly;
                here->DIOcmetal = CONSTepsSiO2 / model->DIOmetalOxideThick  * here->DIOm
                                  * (wm * scale + model->DIOmetalMaskOffset)
                                  * (lm * scale + model->DIOmetalMaskOffset);
                here->DIOcpoly = CONSTepsSiO2 / model->DIOpolyOxideThick  * here->DIOm
                                  * (wp * scale + model->DIOpolyMaskOffset)
                                  * (lp * scale + model->DIOpolyMaskOffset);
            }
            here->DIOforwardKneeCurrent = model->DIOforwardKneeCurrent * here->DIOarea;
            here->DIOreverseKneeCurrent = model->DIOreverseKneeCurrent * here->DIOarea;
            here->DIOjunctionCap = model->DIOjunctionCap * here->DIOarea;
            here->DIOjunctionSWCap = model->DIOjunctionSWCap * here->DIOpj;

            here->DIOstate = *states;
            *states += DIOnumStates;
            if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN) ){
                *states += DIOnumSenStates * (ckt->CKTsenInfo->SENparms);
            }

            if(model->DIOresist == 0) {

                here->DIOposPrimeNode = here->DIOposNode;

            } else if(here->DIOposPrimeNode == 0) {

               CKTnode *tmpNode;
               IFuid tmpName;

                error = CKTmkVolt(ckt,&tmp,here->DIOname,"internal");
                if(error) return(error);
                here->DIOposPrimeNode = tmp->number;
                if (ckt->CKTcopyNodesets) {
                  if (CKTinst2Node(ckt,here,1,&tmpNode,&tmpName)==OK) {
                     if (tmpNode->nsGiven) {
                       tmp->nodeset=tmpNode->nodeset;
                       tmp->nsGiven=tmpNode->nsGiven;
                     }
                  }
                }
            }

            int selfheat = ((here->DIOtempNode > 0) && (here->DIOthermal) && (model->DIOrth0Given));

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

            TSTALLOC(DIOposPosPrimePtr,DIOposNode,DIOposPrimeNode);
            TSTALLOC(DIOnegPosPrimePtr,DIOnegNode,DIOposPrimeNode);
            TSTALLOC(DIOposPrimePosPtr,DIOposPrimeNode,DIOposNode);
            TSTALLOC(DIOposPrimeNegPtr,DIOposPrimeNode,DIOnegNode);
            TSTALLOC(DIOposPosPtr,DIOposNode,DIOposNode);
            TSTALLOC(DIOnegNegPtr,DIOnegNode,DIOnegNode);
            TSTALLOC(DIOposPrimePosPrimePtr,DIOposPrimeNode,DIOposPrimeNode);

            if (selfheat) {
                TSTALLOC(DIOtempPosPtr,      DIOtempNode,     DIOposNode);
                TSTALLOC(DIOtempPosPrimePtr, DIOtempNode,     DIOposPrimeNode);
                TSTALLOC(DIOtempNegPtr,      DIOtempNode,     DIOnegNode);
                TSTALLOC(DIOtempTempPtr,     DIOtempNode,     DIOtempNode);
                TSTALLOC(DIOposTempPtr,      DIOposNode,      DIOtempNode);
                TSTALLOC(DIOposPrimeTempPtr, DIOposPrimeNode, DIOtempNode);
                TSTALLOC(DIOnegTempPtr,      DIOnegNode,      DIOtempNode);
            }

        }
    }
    return(OK);
}

int
DIOunsetup(
    GENmodel *inModel,
    CKTcircuit *ckt)
{
    DIOmodel *model;
    DIOinstance *here;

    for (model = (DIOmodel *)inModel; model != NULL;
        model = DIOnextModel(model))
    {
        for (here = DIOinstances(model); here != NULL;
                here=DIOnextInstance(here))
        {

            if (here->DIOposPrimeNode > 0
              && here->DIOposPrimeNode != here->DIOposNode)
                CKTdltNNum(ckt, here->DIOposPrimeNode);
            here->DIOposPrimeNode = 0;
        }
    }
    return OK;
}

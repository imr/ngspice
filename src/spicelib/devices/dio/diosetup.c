/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
Modified by Dietmar Warning 2003 and Paolo Nenzi 2003
**********/

/* load the diode structure with those pointers needed later 
 * for fast matrix loading 
 */

#include "ngspice.h"
#include "smpdefs.h"
#include "cktdefs.h"
#include "diodefs.h"
#include "sperror.h"
#include "suffix.h"

int
DIOsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
{
    DIOmodel *model = (DIOmodel*)inModel;
    DIOinstance *here;
    int error;
    CKTnode *tmp;

    /*  loop through all the diode models */
    for( ; model != NULL; model = model->DIOnextModel ) {

        if(!model->DIOemissionCoeffGiven) {
            model->DIOemissionCoeff = 1;
        }
        if(!model->DIOsatCurGiven) {
            model->DIOsatCur = 1e-14;
        }
        if(!model->DIOsatSWCurGiven) {
            model->DIOsatSWCur = 0.0;
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
        if(!model->DIOforwardKneeCurrentGiven) {
            model->DIOforwardKneeCurrent = 1e-3;
        } 
        if(!model->DIOreverseKneeCurrentGiven) {
            model->DIOreverseKneeCurrent = 1e-3;
        } 
        if(!model->DIOactivationEnergyGiven) {
            model->DIOactivationEnergy = 1.11;
        } 
        if(!model->DIOsaturationCurrentExpGiven) {
            model->DIOsaturationCurrentExp = 3;
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
        /* loop through all the instances of the model */
        for (here = model->DIOinstances; here != NULL ;
                here=here->DIOnextInstance) {
	    if (here->DIOowner != ARCHme) goto matrixpointers;

            if(!here->DIOareaGiven) {
                here->DIOarea = 1;
            }
            if(!here->DIOpjGiven) {
                here->DIOpj = 0;
            }
            if(!here->DIOmGiven) {
                here->DIOm = 1;
            }

            here->DIOstate = *states;
            *states += 5;
            if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN) ){
                *states += 2 * (ckt->CKTsenInfo->SENparms);
            }
            
matrixpointers:
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

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
if((here->ptr = SMPmakeElt(matrix,here->first,here->second))==(double *)NULL){\
    return(E_NOMEM);\
}

            TSTALLOC(DIOposPosPrimePtr,DIOposNode,DIOposPrimeNode)
            TSTALLOC(DIOnegPosPrimePtr,DIOnegNode,DIOposPrimeNode)
            TSTALLOC(DIOposPrimePosPtr,DIOposPrimeNode,DIOposNode)
            TSTALLOC(DIOposPrimeNegPtr,DIOposPrimeNode,DIOnegNode)
            TSTALLOC(DIOposPosPtr,DIOposNode,DIOposNode)
            TSTALLOC(DIOnegNegPtr,DIOnegNode,DIOnegNode)
            TSTALLOC(DIOposPrimePosPrimePtr,DIOposPrimeNode,DIOposPrimeNode)
        }
    }
    return(OK);
}

int
DIOunsetup(inModel,ckt)
    GENmodel *inModel;
    CKTcircuit *ckt;
{
    DIOmodel *model;
    DIOinstance *here;

    for (model = (DIOmodel *)inModel; model != NULL;
	    model = model->DIOnextModel)
    {
        for (here = model->DIOinstances; here != NULL;
                here=here->DIOnextInstance)
	{

	    if (here->DIOposPrimeNode
		    && here->DIOposPrimeNode != here->DIOposNode)
	    {
		CKTdltNNum(ckt, here->DIOposPrimeNode);
		here->DIOposPrimeNode = 0;
	    }
	}
    }
    return OK;
}

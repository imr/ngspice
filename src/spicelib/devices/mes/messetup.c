/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 S. Hwang
Modified: 2000 AlansFixes
**********/

#include "ngspice.h"
#include "smpdefs.h"
#include "cktdefs.h"
#include "mesdefs.h"
#include "const.h"
#include "sperror.h"
#include "suffix.h"

int
MESsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
        /* load the diode structure with those pointers needed later 
         * for fast matrix loading 
         */
{
    MESmodel *model = (MESmodel*)inModel;
    MESinstance *here;
    int error;
    CKTnode *tmp;

    /*  loop through all the diode models */
    for( ; model != NULL; model = model->MESnextModel ) {

        if( (model->MEStype != NMF) && (model->MEStype != PMF) ) {
            model->MEStype = NMF;
        }
        if(!model->MESthresholdGiven) {
            model->MESthreshold = -2;
        }
        if(!model->MESbetaGiven) {
            model->MESbeta = 2.5e-3;
        }
        if(!model->MESbGiven) {
            model->MESb = 0.3;
        }
        if(!model->MESalphaGiven) {
            model->MESalpha = 2;
        }
        if(!model->MESlModulationGiven) {
            model->MESlModulation = 0;
        }
        if(!model->MESdrainResistGiven) {
            model->MESdrainResist = 0;
        }
        if(!model->MESsourceResistGiven) {
            model->MESsourceResist = 0;
        }
        if(!model->MEScapGSGiven) {
            model->MEScapGS = 0;
        }
        if(!model->MEScapGDGiven) {
            model->MEScapGD = 0;
        }
        if(!model->MESgatePotentialGiven) {
            model->MESgatePotential = 1;
        }
        if(!model->MESgateSatCurrentGiven) {
            model->MESgateSatCurrent = 1e-14;
        }
        if(!model->MESdepletionCapCoeffGiven) {
            model->MESdepletionCapCoeff = .5;
        }
	if(!model->MESfNcoefGiven) {
	    model->MESfNcoef = 0;
	}
	if(!model->MESfNexpGiven) {
	    model->MESfNexp = 1;
	}

        /* loop through all the instances of the model */
        for (here = model->MESinstances; here != NULL ;
                here=here->MESnextInstance) {
	    if (here->MESowner != ARCHme) goto matrixpointers;
            
            if(!here->MESareaGiven) {
                here->MESarea = 1.0;
            }
            if(!here->MESmGiven) {
                here->MESm = 1.0;
            }
            here->MESstate = *states;
            *states += MESnumStates;

matrixpointers:
            if(model->MESsourceResist != 0 && here->MESsourcePrimeNode==0) {
                error = CKTmkVolt(ckt,&tmp,here->MESname,"source");
                if(error) return(error);
                here->MESsourcePrimeNode = tmp->number;
                
                if (ckt->CKTcopyNodesets) {
		    CKTnode *tmpNode;
		    IFuid tmpName;

                  if (CKTinst2Node(ckt,here,3,&tmpNode,&tmpName)==OK) {
                     if (tmpNode->nsGiven) {
                       tmp->nodeset=tmpNode->nodeset; 
                       tmp->nsGiven=tmpNode->nsGiven; 
                     }
                  }
                }
                
            } else {
                here->MESsourcePrimeNode = here->MESsourceNode;
            }
            if(model->MESdrainResist != 0 && here->MESdrainPrimeNode==0) {
                error = CKTmkVolt(ckt,&tmp,here->MESname,"drain");
                if(error) return(error);
                here->MESdrainPrimeNode = tmp->number;
                
                if (ckt->CKTcopyNodesets) {
		    CKTnode *tmpNode;
		    IFuid tmpName;

                  if (CKTinst2Node(ckt,here,1,&tmpNode,&tmpName)==OK) {
                     if (tmpNode->nsGiven) {
                       tmp->nodeset=tmpNode->nodeset; 
                       tmp->nsGiven=tmpNode->nsGiven; 
                     }
                  }
                }
                
            } else {
                here->MESdrainPrimeNode = here->MESdrainNode;
            }

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
if((here->ptr = SMPmakeElt(matrix,here->first,here->second))==(double *)NULL){\
    return(E_NOMEM);\
}

            TSTALLOC(MESdrainDrainPrimePtr,MESdrainNode,MESdrainPrimeNode)
            TSTALLOC(MESgateDrainPrimePtr,MESgateNode,MESdrainPrimeNode)
            TSTALLOC(MESgateSourcePrimePtr,MESgateNode,MESsourcePrimeNode)
            TSTALLOC(MESsourceSourcePrimePtr,MESsourceNode,
                    MESsourcePrimeNode)
            TSTALLOC(MESdrainPrimeDrainPtr,MESdrainPrimeNode,MESdrainNode)
            TSTALLOC(MESdrainPrimeGatePtr,MESdrainPrimeNode,MESgateNode)
            TSTALLOC(MESdrainPrimeSourcePrimePtr,MESdrainPrimeNode,
                    MESsourcePrimeNode)
            TSTALLOC(MESsourcePrimeGatePtr,MESsourcePrimeNode,MESgateNode)
            TSTALLOC(MESsourcePrimeSourcePtr,MESsourcePrimeNode,
                    MESsourceNode)
            TSTALLOC(MESsourcePrimeDrainPrimePtr,MESsourcePrimeNode,
                    MESdrainPrimeNode)
            TSTALLOC(MESdrainDrainPtr,MESdrainNode,MESdrainNode)
            TSTALLOC(MESgateGatePtr,MESgateNode,MESgateNode)
            TSTALLOC(MESsourceSourcePtr,MESsourceNode,MESsourceNode)
            TSTALLOC(MESdrainPrimeDrainPrimePtr,MESdrainPrimeNode,
                    MESdrainPrimeNode)
            TSTALLOC(MESsourcePrimeSourcePrimePtr,MESsourcePrimeNode,
                    MESsourcePrimeNode)
        }
    }
    return(OK);
}

int
MESunsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    MESmodel *model;
    MESinstance *here;

    for (model = (MESmodel *)inModel; model != NULL;
	    model = model->MESnextModel)
    {
        for (here = model->MESinstances; here != NULL;
                here=here->MESnextInstance)
	{
	    if (here->MESsourcePrimeNode
		    && here->MESsourcePrimeNode != here->MESsourceNode)
	    {
		CKTdltNNum(ckt, here->MESsourcePrimeNode);
		here->MESsourcePrimeNode = 0;
	    }
	    if (here->MESdrainPrimeNode
		    && here->MESdrainPrimeNode != here->MESdrainNode)
	    {
		CKTdltNNum(ckt, here->MESdrainPrimeNode);
		here->MESdrainPrimeNode = 0;
	    }
	}
    }
    return OK;
}

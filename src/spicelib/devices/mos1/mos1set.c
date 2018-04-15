/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

    /* load the MOS1 device structure with those pointers needed later 
     * for fast matrix loading 
     */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "mos1defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
MOS1setup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt,
          int *states)
{
    MOS1model *model = (MOS1model *)inModel;
    MOS1instance *here;
    int error;
    CKTnode *tmp;

    /*  loop through all the MOS1 device models */
    for( ; model != NULL; model = MOS1nextModel(model)) {

        if(!model->MOS1typeGiven) {
            model->MOS1type = NMOS;
        }
        if(!model->MOS1latDiffGiven) {
            model->MOS1latDiff = 0;
        }
        if(!model->MOS1jctSatCurDensityGiven) {
            model->MOS1jctSatCurDensity = 0;
        }
        if(!model->MOS1jctSatCurGiven) {
            model->MOS1jctSatCur = 1e-14;
        }
        if(!model->MOS1transconductanceGiven) {
            model->MOS1transconductance = 2e-5;
        }
        if(!model->MOS1gateSourceOverlapCapFactorGiven) {
            model->MOS1gateSourceOverlapCapFactor = 0;
        }
        if(!model->MOS1gateDrainOverlapCapFactorGiven) {
            model->MOS1gateDrainOverlapCapFactor = 0;
        }
        if(!model->MOS1gateBulkOverlapCapFactorGiven) {
            model->MOS1gateBulkOverlapCapFactor = 0;
        }
        if(!model->MOS1vt0Given) {
            model->MOS1vt0 = 0;
        }
        if(!model->MOS1bulkCapFactorGiven) {
            model->MOS1bulkCapFactor = 0;
        }
        if(!model->MOS1sideWallCapFactorGiven) {
            model->MOS1sideWallCapFactor = 0;
        }
        if(!model->MOS1bulkJctPotentialGiven) {
            model->MOS1bulkJctPotential = .8;
        }
        if(!model->MOS1bulkJctBotGradingCoeffGiven) {
            model->MOS1bulkJctBotGradingCoeff = .5;
        }
        if(!model->MOS1bulkJctSideGradingCoeffGiven) {
            model->MOS1bulkJctSideGradingCoeff = .5;
        }
        if(!model->MOS1fwdCapDepCoeffGiven) {
            model->MOS1fwdCapDepCoeff = .5;
        }
        if(!model->MOS1phiGiven) {
            model->MOS1phi = .6;
        }
        if(!model->MOS1lambdaGiven) {
            model->MOS1lambda = 0;
        }
        if(!model->MOS1gammaGiven) {
            model->MOS1gamma = 0;
        }
	if(!model->MOS1fNcoefGiven) {
	    model->MOS1fNcoef = 0;
	}
	if(!model->MOS1fNexpGiven) {
	    model->MOS1fNexp = 1;
	}

        /* loop through all the instances of the model */
        for (here = MOS1instances(model); here != NULL ;
                here=MOS1nextInstance(here)) {

            /* allocate a chunk of the state vector */
            here->MOS1states = *states;
            *states += MOS1numStates;

            if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN) ){
                *states += MOS1numSenStates * (ckt->CKTsenInfo->SENparms);
            }

            if(!here->MOS1drainPerimiterGiven) {
                here->MOS1drainPerimiter = 0;
            }
            if(!here->MOS1icVBSGiven) {
                here->MOS1icVBS = 0;
            }
            if(!here->MOS1icVDSGiven) {
                here->MOS1icVDS = 0;
            }
            if(!here->MOS1icVGSGiven) {
                here->MOS1icVGS = 0;
            }
            if(!here->MOS1sourcePerimiterGiven) {
                here->MOS1sourcePerimiter = 0;
            }
            if(!here->MOS1vdsatGiven) {
                here->MOS1vdsat = 0;
            }
            if(!here->MOS1vonGiven) {
                here->MOS1von = 0;
            }
	    if(!here->MOS1drainSquaresGiven) {
		here->MOS1drainSquares=1;
	    }
	    if(!here->MOS1sourceSquaresGiven) {
		here->MOS1sourceSquares=1;
	    }

            if ((model->MOS1drainResistance != 0
		    || (model->MOS1sheetResistance != 0
                    && here->MOS1drainSquares != 0) )) {
                if (here->MOS1dNodePrime == 0) {
                error = CKTmkVolt(ckt,&tmp,here->MOS1name,"drain");
                if(error) return(error);
                here->MOS1dNodePrime = tmp->number;
                
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
                }
                
            } else {
                here->MOS1dNodePrime = here->MOS1dNode;
            }

            if((model->MOS1sourceResistance != 0 ||
                    (model->MOS1sheetResistance != 0 &&
                     here->MOS1sourceSquares != 0) )) {
                if (here->MOS1sNodePrime == 0) {
                error = CKTmkVolt(ckt,&tmp,here->MOS1name,"source");
                if(error) return(error);
                here->MOS1sNodePrime = tmp->number;
                
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
                
                }
            } else {
                here->MOS1sNodePrime = here->MOS1sNode;
            }

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)
            TSTALLOC(MOS1DdPtr,MOS1dNode,MOS1dNode);
            TSTALLOC(MOS1GgPtr,MOS1gNode,MOS1gNode);
            TSTALLOC(MOS1SsPtr,MOS1sNode,MOS1sNode);
            TSTALLOC(MOS1BbPtr,MOS1bNode,MOS1bNode);
            TSTALLOC(MOS1DPdpPtr,MOS1dNodePrime,MOS1dNodePrime);
            TSTALLOC(MOS1SPspPtr,MOS1sNodePrime,MOS1sNodePrime);
            TSTALLOC(MOS1DdpPtr,MOS1dNode,MOS1dNodePrime);
            TSTALLOC(MOS1GbPtr,MOS1gNode,MOS1bNode);
            TSTALLOC(MOS1GdpPtr,MOS1gNode,MOS1dNodePrime);
            TSTALLOC(MOS1GspPtr,MOS1gNode,MOS1sNodePrime);
            TSTALLOC(MOS1SspPtr,MOS1sNode,MOS1sNodePrime);
            TSTALLOC(MOS1BdpPtr,MOS1bNode,MOS1dNodePrime);
            TSTALLOC(MOS1BspPtr,MOS1bNode,MOS1sNodePrime);
            TSTALLOC(MOS1DPspPtr,MOS1dNodePrime,MOS1sNodePrime);
            TSTALLOC(MOS1DPdPtr,MOS1dNodePrime,MOS1dNode);
            TSTALLOC(MOS1BgPtr,MOS1bNode,MOS1gNode);
            TSTALLOC(MOS1DPgPtr,MOS1dNodePrime,MOS1gNode);
            TSTALLOC(MOS1SPgPtr,MOS1sNodePrime,MOS1gNode);
            TSTALLOC(MOS1SPsPtr,MOS1sNodePrime,MOS1sNode);
            TSTALLOC(MOS1DPbPtr,MOS1dNodePrime,MOS1bNode);
            TSTALLOC(MOS1SPbPtr,MOS1sNodePrime,MOS1bNode);
            TSTALLOC(MOS1SPdpPtr,MOS1sNodePrime,MOS1dNodePrime);

        }
    }
    return(OK);
}

int
MOS1unsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS1model *model;
    MOS1instance *here;

    for (model = (MOS1model *)inModel; model != NULL;
	    model = MOS1nextModel(model))
    {
        for (here = MOS1instances(model); here != NULL;
                here=MOS1nextInstance(here))
	{
	    if (here->MOS1sNodePrime > 0
		    && here->MOS1sNodePrime != here->MOS1sNode)
		CKTdltNNum(ckt, here->MOS1sNodePrime);
            here->MOS1sNodePrime= 0;

	    if (here->MOS1dNodePrime > 0
		    && here->MOS1dNodePrime != here->MOS1dNode)
		CKTdltNNum(ckt, here->MOS1dNodePrime);
            here->MOS1dNodePrime= 0;
	}
    }
    return OK;
}

/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1989 Takayasu Sakurai
Modified: 2000 AlansFixes
**********/

    /* load the MOS6 device structure with those pointers needed later 
     * for fast matrix loading 
     */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "mos6defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
MOS6setup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt,
          int *states)
{
    MOS6model *model = (MOS6model *)inModel;
    MOS6instance *here;
    int error;
    CKTnode *tmp;

    /*  loop through all the MOS6 device models */
    for( ; model != NULL; model = MOS6nextModel(model)) {

        if(!model->MOS6typeGiven) {
            model->MOS6type = NMOS;
        }
        if(!model->MOS6latDiffGiven) {
            model->MOS6latDiff = 0;
        }
        if(!model->MOS6jctSatCurDensityGiven) {
            model->MOS6jctSatCurDensity = 0;
        }
        if(!model->MOS6jctSatCurGiven) {
            model->MOS6jctSatCur = 1e-14;
        }
        if(!model->MOS6kvGiven) {
            model->MOS6kv = 2;
        }
        if(!model->MOS6nvGiven) {
            model->MOS6nv = 0.5;
        }
        if(!model->MOS6kcGiven) {
            model->MOS6kc = 5e-5;
        }
        if(!model->MOS6ncGiven) {
            model->MOS6nc = 1;
        }
        if(!model->MOS6nvthGiven) {
            model->MOS6nvth = 0.5;
        }
        if(!model->MOS6psGiven) {
            model->MOS6ps = 0;
        }
        if(!model->MOS6gateSourceOverlapCapFactorGiven) {
            model->MOS6gateSourceOverlapCapFactor = 0;
        }
        if(!model->MOS6gateDrainOverlapCapFactorGiven) {
            model->MOS6gateDrainOverlapCapFactor = 0;
        }
        if(!model->MOS6gateBulkOverlapCapFactorGiven) {
            model->MOS6gateBulkOverlapCapFactor = 0;
        }
        if(!model->MOS6vt0Given) {
            model->MOS6vt0 = 0;
        }
        if(!model->MOS6bulkCapFactorGiven) {
            model->MOS6bulkCapFactor = 0;
        }
        if(!model->MOS6sideWallCapFactorGiven) {
            model->MOS6sideWallCapFactor = 0;
        }
        if(!model->MOS6bulkJctPotentialGiven) {
            model->MOS6bulkJctPotential = .8;
        }
        if(!model->MOS6bulkJctBotGradingCoeffGiven) {
            model->MOS6bulkJctBotGradingCoeff = .5;
        }
        if(!model->MOS6bulkJctSideGradingCoeffGiven) {
            model->MOS6bulkJctSideGradingCoeff = .5;
        }
        if(!model->MOS6fwdCapDepCoeffGiven) {
            model->MOS6fwdCapDepCoeff = .5;
        }
        if(!model->MOS6phiGiven) {
            model->MOS6phi = .6;
        }
        if(!model->MOS6lamda0Given) {
            model->MOS6lamda0 = 0;
            if(model->MOS6lambdaGiven) {
                model->MOS6lamda0 = model->MOS6lambda;
	    }
        }
        if(!model->MOS6lamda1Given) {
            model->MOS6lamda1 = 0;
        }
        if(!model->MOS6sigmaGiven) {
            model->MOS6sigma = 0;
        }
        if(!model->MOS6gammaGiven) {
            model->MOS6gamma = 0;
        }
        if(!model->MOS6gamma1Given) {
            model->MOS6gamma1 = 0;
        }

        /* loop through all the instances of the model */
        for (here = MOS6instances(model); here != NULL ;
                here=MOS6nextInstance(here)) {
            CKTnode *tmpNode;
            IFuid tmpName;

            if(!here->MOS6drainPerimiterGiven) {
                here->MOS6drainPerimiter = 0;
            }
            if(!here->MOS6icVBSGiven) {
                here->MOS6icVBS = 0;
            }
            if(!here->MOS6icVDSGiven) {
                here->MOS6icVDS = 0;
            }
            if(!here->MOS6icVGSGiven) {
                here->MOS6icVGS = 0;
            }
            if(!here->MOS6sourcePerimiterGiven) {
                here->MOS6sourcePerimiter = 0;
            }
            if(!here->MOS6vdsatGiven) {
                here->MOS6vdsat = 0;
            }
            if(!here->MOS6vonGiven) {
                here->MOS6von = 0;
            }
            if(!here->MOS6mGiven) {
                here->MOS6m = 1;
            }


            /* allocate a chunk of the state vector */
            here->MOS6states = *states;
            *states += MOS6numStates;
            if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN) ){
                *states += MOS6numSenStates * (ckt->CKTsenInfo->SENparms);
            }

            if((model->MOS6drainResistance != 0 ||
                    (model->MOS6sheetResistance != 0 &&
                     here->MOS6drainSquares != 0) )) {
                if (here->MOS6dNodePrime == 0) {
                error = CKTmkVolt(ckt,&tmp,here->MOS6name,"drain");
                if(error) return(error);
                here->MOS6dNodePrime = tmp->number;
                if (ckt->CKTcopyNodesets) {
                  if (CKTinst2Node(ckt,here,1,&tmpNode,&tmpName)==OK) {
                     if (tmpNode->nsGiven) {
                       tmp->nodeset=tmpNode->nodeset; 
                       tmp->nsGiven=tmpNode->nsGiven; 
                     }
                  }
                }
                }
            } else {
                here->MOS6dNodePrime = here->MOS6dNode;
            }

            if((model->MOS6sourceResistance != 0 ||
                    (model->MOS6sheetResistance != 0 &&
                     here->MOS6sourceSquares != 0) )) {
                if (here->MOS6sNodePrime == 0) {
                error = CKTmkVolt(ckt,&tmp,here->MOS6name,"source");
                if(error) return(error);
                here->MOS6sNodePrime = tmp->number;
                if (ckt->CKTcopyNodesets) {
                  if (CKTinst2Node(ckt,here,3,&tmpNode,&tmpName)==OK) {
                     if (tmpNode->nsGiven) {
                       tmp->nodeset=tmpNode->nodeset; 
                       tmp->nsGiven=tmpNode->nsGiven; 
                     }
                  }
                }
                }
            } else {
                here->MOS6sNodePrime = here->MOS6sNode;
            }
/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

            TSTALLOC(MOS6DdPtr,MOS6dNode,MOS6dNode);
            TSTALLOC(MOS6GgPtr,MOS6gNode,MOS6gNode);
            TSTALLOC(MOS6SsPtr,MOS6sNode,MOS6sNode);
            TSTALLOC(MOS6BbPtr,MOS6bNode,MOS6bNode);
            TSTALLOC(MOS6DPdpPtr,MOS6dNodePrime,MOS6dNodePrime);
            TSTALLOC(MOS6SPspPtr,MOS6sNodePrime,MOS6sNodePrime);
            TSTALLOC(MOS6DdpPtr,MOS6dNode,MOS6dNodePrime);
            TSTALLOC(MOS6GbPtr,MOS6gNode,MOS6bNode);
            TSTALLOC(MOS6GdpPtr,MOS6gNode,MOS6dNodePrime);
            TSTALLOC(MOS6GspPtr,MOS6gNode,MOS6sNodePrime);
            TSTALLOC(MOS6SspPtr,MOS6sNode,MOS6sNodePrime);
            TSTALLOC(MOS6BdpPtr,MOS6bNode,MOS6dNodePrime);
            TSTALLOC(MOS6BspPtr,MOS6bNode,MOS6sNodePrime);
            TSTALLOC(MOS6DPspPtr,MOS6dNodePrime,MOS6sNodePrime);
            TSTALLOC(MOS6DPdPtr,MOS6dNodePrime,MOS6dNode);
            TSTALLOC(MOS6BgPtr,MOS6bNode,MOS6gNode);
            TSTALLOC(MOS6DPgPtr,MOS6dNodePrime,MOS6gNode);
            TSTALLOC(MOS6SPgPtr,MOS6sNodePrime,MOS6gNode);
            TSTALLOC(MOS6SPsPtr,MOS6sNodePrime,MOS6sNode);
            TSTALLOC(MOS6DPbPtr,MOS6dNodePrime,MOS6bNode);
            TSTALLOC(MOS6SPbPtr,MOS6sNodePrime,MOS6bNode);
            TSTALLOC(MOS6SPdpPtr,MOS6sNodePrime,MOS6dNodePrime);

        }
    }
    return(OK);
}

int
MOS6unsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS6model *model;
    MOS6instance *here;

    for (model = (MOS6model *)inModel; model != NULL;
	    model = MOS6nextModel(model))
    {
        for (here = MOS6instances(model); here != NULL;
                here=MOS6nextInstance(here))
	{
	    if (here->MOS6sNodePrime > 0
		    && here->MOS6sNodePrime != here->MOS6sNode)
		CKTdltNNum(ckt, here->MOS6sNodePrime);
            here->MOS6sNodePrime= 0;

	    if (here->MOS6dNodePrime > 0
		    && here->MOS6dNodePrime != here->MOS6dNode)
		CKTdltNNum(ckt, here->MOS6dNodePrime);
            here->MOS6dNodePrime= 0;
	}
    }
    return OK;
}

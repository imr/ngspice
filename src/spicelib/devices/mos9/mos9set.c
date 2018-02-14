/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "mos9defs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/* assuming silicon - make definition for epsilon of silicon */
#define EPSSIL (11.7 * 8.854214871e-12)

int
MOS9setup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
        /* load the MOS9 device structure with those pointers needed later 
         * for fast matrix loading 
         */

{
    register MOS9model *model = (MOS9model *)inModel;
    register MOS9instance *here;
    int error;
    CKTnode *tmp;

    /*  loop through all the MOS9 device models */
    for( ; model != NULL; model = MOS9nextModel(model)) {

        /* perform model defaulting */
        if(!model->MOS9typeGiven) {
            model->MOS9type = NMOS;
        }
        if(!model->MOS9latDiffGiven) {
            model->MOS9latDiff = 0;
        }
        if(!model->MOS9lengthAdjustGiven) {
            model->MOS9lengthAdjust = 0;
        }
        if(!model->MOS9widthNarrowGiven) {
            model->MOS9widthNarrow = 0;
        }
        if(!model->MOS9widthAdjustGiven) {
            model->MOS9widthAdjust = 0;
        }
        if(!model->MOS9delvt0Given) {
            model->MOS9delvt0 = 0;
        }
        if(!model->MOS9jctSatCurDensityGiven) {
            model->MOS9jctSatCurDensity = 0;
        }
        if(!model->MOS9jctSatCurGiven) {
            model->MOS9jctSatCur = 1e-14;
        }
        if(!model->MOS9drainResistanceGiven) {
            model->MOS9drainResistance = 0;
        }
        if(!model->MOS9sourceResistanceGiven) {
            model->MOS9sourceResistance = 0;
        }
        if(!model->MOS9sheetResistanceGiven) {
            model->MOS9sheetResistance = 0;
        }
        if(!model->MOS9transconductanceGiven) {
            model->MOS9transconductance = 2e-5;
        }
        if(!model->MOS9gateSourceOverlapCapFactorGiven) {
            model->MOS9gateSourceOverlapCapFactor = 0;
        }
        if(!model->MOS9gateDrainOverlapCapFactorGiven) {
            model->MOS9gateDrainOverlapCapFactor = 0;
        }
        if(!model->MOS9gateBulkOverlapCapFactorGiven) {
            model->MOS9gateBulkOverlapCapFactor = 0;
        }
        if(!model->MOS9vt0Given) {
            model->MOS9vt0 = 0;
        }
        if(!model->MOS9capBDGiven) {
            model->MOS9capBD = 0;
        }
        if(!model->MOS9capBSGiven) {
            model->MOS9capBS = 0;
        }
        if(!model->MOS9bulkCapFactorGiven) {
            model->MOS9bulkCapFactor = 0;
        }
        if(!model->MOS9sideWallCapFactorGiven) {
            model->MOS9sideWallCapFactor = 0;
        }
        if(!model->MOS9bulkJctPotentialGiven) {
            model->MOS9bulkJctPotential = .8;
        }
        if(!model->MOS9bulkJctBotGradingCoeffGiven) {
            model->MOS9bulkJctBotGradingCoeff = .5;
        }
        if(!model->MOS9bulkJctSideGradingCoeffGiven) {
            model->MOS9bulkJctSideGradingCoeff = .33;
        }
        if(!model->MOS9fwdCapDepCoeffGiven) {
            model->MOS9fwdCapDepCoeff = .5;
        }
        if(!model->MOS9phiGiven) {
            model->MOS9phi = .6;
        }
        if(!model->MOS9gammaGiven) {
            model->MOS9gamma = 0;
        }
        if(!model->MOS9deltaGiven) {
            model->MOS9delta = 0;
        }
        if(!model->MOS9maxDriftVelGiven) {
            model->MOS9maxDriftVel = 0;
        }
        if(!model->MOS9junctionDepthGiven) {
            model->MOS9junctionDepth = 0;
        }
        if(!model->MOS9fastSurfaceStateDensityGiven) {
            model->MOS9fastSurfaceStateDensity = 0;
        }
        if(!model->MOS9etaGiven) {
            model->MOS9eta = 0;
        }
        if(!model->MOS9thetaGiven) {
            model->MOS9theta = 0;
        }
        if(!model->MOS9kappaGiven) {
            model->MOS9kappa = .2;
        }
        if(!model->MOS9oxideThicknessGiven) {
            model->MOS9oxideThickness = 1e-7;
        } 
	if(!model->MOS9fNcoefGiven) {
	    model->MOS9fNcoef = 0;
	}
	if(!model->MOS9fNexpGiven) {
	    model->MOS9fNexp = 1;
	}

        /* loop through all the instances of the model */
        for (here = MOS9instances(model); here != NULL ;
                here=MOS9nextInstance(here)) {

            CKTnode *tmpNode;
            IFuid tmpName;

            /* allocate a chunk of the state vector */
            here->MOS9states = *states;
            *states += MOS9NUMSTATES;

            if(!here->MOS9drainAreaGiven) {
                here->MOS9drainArea = ckt->CKTdefaultMosAD;
            }
            if(!here->MOS9drainPerimiterGiven) {
                here->MOS9drainPerimiter = 0;
            }
            if(!here->MOS9drainSquaresGiven) {
                here->MOS9drainSquares = 1;
            }
            if(!here->MOS9icVBSGiven) {
                here->MOS9icVBS = 0;
            }
            if(!here->MOS9icVDSGiven) {
                here->MOS9icVDS = 0;
            }
            if(!here->MOS9icVGSGiven) {
                here->MOS9icVGS = 0;
            }
            if(!here->MOS9sourcePerimiterGiven) {
                here->MOS9sourcePerimiter = 0;
            }
            if(!here->MOS9sourceSquaresGiven) {
                here->MOS9sourceSquares = 1;
            }
            if(!here->MOS9vdsatGiven) {
                here->MOS9vdsat = 0;
            }
            if(!here->MOS9vonGiven) {
                here->MOS9von = 0;
            }
            if(!here->MOS9modeGiven) {
                here->MOS9mode = 1;
            }

            if((model->MOS9drainResistance != 0 ||
                    (model->MOS9sheetResistance != 0 &&
                     here->MOS9drainSquares != 0      ) )) {
                if (here->MOS9dNodePrime==0) {
                error = CKTmkVolt(ckt,&tmp,here->MOS9name,"internal#drain");
                if(error) return(error);
                here->MOS9dNodePrime = tmp->number;
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
                here->MOS9dNodePrime = here->MOS9dNode;
            }

            if((model->MOS9sourceResistance != 0 ||
                    (model->MOS9sheetResistance != 0 && 
                     here->MOS9sourceSquares != 0     ) )) {
                if (here->MOS9sNodePrime == 0) {
                error = CKTmkVolt(ckt,&tmp,here->MOS9name,"internal#source");
                if(error) return(error);
                here->MOS9sNodePrime = tmp->number;
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
                here->MOS9sNodePrime = here->MOS9sNode;
            }

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

            TSTALLOC(MOS9DdPtr, MOS9dNode, MOS9dNode);
            TSTALLOC(MOS9GgPtr, MOS9gNode, MOS9gNode);
            TSTALLOC(MOS9SsPtr, MOS9sNode, MOS9sNode);
            TSTALLOC(MOS9BbPtr, MOS9bNode, MOS9bNode);
            TSTALLOC(MOS9DPdpPtr, MOS9dNodePrime, MOS9dNodePrime);
            TSTALLOC(MOS9SPspPtr, MOS9sNodePrime, MOS9sNodePrime);
            TSTALLOC(MOS9DdpPtr, MOS9dNode, MOS9dNodePrime);
            TSTALLOC(MOS9GbPtr, MOS9gNode, MOS9bNode);
            TSTALLOC(MOS9GdpPtr, MOS9gNode, MOS9dNodePrime);
            TSTALLOC(MOS9GspPtr, MOS9gNode, MOS9sNodePrime);
            TSTALLOC(MOS9SspPtr, MOS9sNode, MOS9sNodePrime);
            TSTALLOC(MOS9BdpPtr, MOS9bNode, MOS9dNodePrime);
            TSTALLOC(MOS9BspPtr, MOS9bNode, MOS9sNodePrime);
            TSTALLOC(MOS9DPspPtr, MOS9dNodePrime, MOS9sNodePrime);
            TSTALLOC(MOS9DPdPtr, MOS9dNodePrime, MOS9dNode);
            TSTALLOC(MOS9BgPtr, MOS9bNode, MOS9gNode);
            TSTALLOC(MOS9DPgPtr, MOS9dNodePrime, MOS9gNode);
            TSTALLOC(MOS9SPgPtr, MOS9sNodePrime, MOS9gNode);
            TSTALLOC(MOS9SPsPtr, MOS9sNodePrime, MOS9sNode);
            TSTALLOC(MOS9DPbPtr, MOS9dNodePrime, MOS9bNode);
            TSTALLOC(MOS9SPbPtr, MOS9sNodePrime, MOS9bNode);
            TSTALLOC(MOS9SPdpPtr, MOS9sNodePrime, MOS9dNodePrime);

        }
    }
    return(OK);
}

int
MOS9unsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS9model *model;
    MOS9instance *here;

    for (model = (MOS9model *)inModel; model != NULL;
	    model = MOS9nextModel(model))
    {
        for (here = MOS9instances(model); here != NULL;
                here=MOS9nextInstance(here))
	{
	    if (here->MOS9sNodePrime > 0
		    && here->MOS9sNodePrime != here->MOS9sNode)
		CKTdltNNum(ckt, here->MOS9sNodePrime);
            here->MOS9sNodePrime= 0;

	    if (here->MOS9dNodePrime > 0
		    && here->MOS9dNodePrime != here->MOS9dNode)
		CKTdltNNum(ckt, here->MOS9dNodePrime);
            here->MOS9dNodePrime= 0;
	}
    }
    return OK;
}

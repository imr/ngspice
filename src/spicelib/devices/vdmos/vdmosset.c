/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

    /* load the VDMOS device structure with those pointers needed later 
     * for fast matrix loading 
     */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "vdmosdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
VDMOSsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt,
          int *states)
{
    VDMOSmodel *model = (VDMOSmodel *)inModel;
    VDMOSinstance *here;
    int error;
    CKTnode *tmp;

    /*  loop through all the VDMOS device models */
    for( ; model != NULL; model = VDMOSnextModel(model)) {

        if(!model->VDMOStypeGiven) {
            model->VDMOStype = NMOS;
        }
        if(!model->VDMOSlatDiffGiven) {
            model->VDMOSlatDiff = 0;
        }
        if(!model->VDMOSjctSatCurDensityGiven) {
            model->VDMOSjctSatCurDensity = 0;
        }
        if(!model->VDMOSjctSatCurGiven) {
            model->VDMOSjctSatCur = 1e-14;
        }
        if(!model->VDMOStransconductanceGiven) {
            model->VDMOStransconductance = 2e-5;
        }
        if(!model->VDMOSgateSourceOverlapCapFactorGiven) {
            model->VDMOSgateSourceOverlapCapFactor = 0;
        }
        if(!model->VDMOSgateDrainOverlapCapFactorGiven) {
            model->VDMOSgateDrainOverlapCapFactor = 0;
        }
        if(!model->VDMOSgateBulkOverlapCapFactorGiven) {
            model->VDMOSgateBulkOverlapCapFactor = 0;
        }
        if(!model->VDMOSvt0Given) {
            model->VDMOSvt0 = 0;
        }
        if(!model->VDMOSbulkCapFactorGiven) {
            model->VDMOSbulkCapFactor = 0;
        }
        if(!model->VDMOSsideWallCapFactorGiven) {
            model->VDMOSsideWallCapFactor = 0;
        }
        if(!model->VDMOSbulkJctPotentialGiven) {
            model->VDMOSbulkJctPotential = .8;
        }
        if(!model->VDMOSbulkJctBotGradingCoeffGiven) {
            model->VDMOSbulkJctBotGradingCoeff = .5;
        }
        if(!model->VDMOSbulkJctSideGradingCoeffGiven) {
            model->VDMOSbulkJctSideGradingCoeff = .5;
        }
        if(!model->VDMOSfwdCapDepCoeffGiven) {
            model->VDMOSfwdCapDepCoeff = .5;
        }
        if(!model->VDMOSphiGiven) {
            model->VDMOSphi = .6;
        }
        if(!model->VDMOSlambdaGiven) {
            model->VDMOSlambda = 0;
        }
        if(!model->VDMOSgammaGiven) {
            model->VDMOSgamma = 0;
        }
	if(!model->VDMOSfNcoefGiven) {
	    model->VDMOSfNcoef = 0;
	}
	if(!model->VDMOSfNexpGiven) {
	    model->VDMOSfNexp = 1;
	}

        /* loop through all the instances of the model */
        for (here = VDMOSinstances(model); here != NULL ;
                here=VDMOSnextInstance(here)) {

            /* allocate a chunk of the state vector */
            here->VDMOSstates = *states;
            *states += VDMOSnumStates;

            if(!here->VDMOSdrainPerimiterGiven) {
                here->VDMOSdrainPerimiter = 0;
            }
            if(!here->VDMOSicVBSGiven) {
                here->VDMOSicVBS = 0;
            }
            if(!here->VDMOSicVDSGiven) {
                here->VDMOSicVDS = 0;
            }
            if(!here->VDMOSicVGSGiven) {
                here->VDMOSicVGS = 0;
            }
            if(!here->VDMOSsourcePerimiterGiven) {
                here->VDMOSsourcePerimiter = 0;
            }
            if(!here->VDMOSvdsatGiven) {
                here->VDMOSvdsat = 0;
            }
            if(!here->VDMOSvonGiven) {
                here->VDMOSvon = 0;
            }
	    if(!here->VDMOSdrainSquaresGiven) {
		here->VDMOSdrainSquares=1;
	    }
	    if(!here->VDMOSsourceSquaresGiven) {
		here->VDMOSsourceSquares=1;
	    }

            if ((model->VDMOSdrainResistance != 0
		    || (model->VDMOSsheetResistance != 0
                    && here->VDMOSdrainSquares != 0) )) {
                if (here->VDMOSdNodePrime == 0) {
                error = CKTmkVolt(ckt,&tmp,here->VDMOSname,"drain");
                if(error) return(error);
                here->VDMOSdNodePrime = tmp->number;
                
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
                here->VDMOSdNodePrime = here->VDMOSdNode;
            }

            if((model->VDMOSsourceResistance != 0 ||
                    (model->VDMOSsheetResistance != 0 &&
                     here->VDMOSsourceSquares != 0) )) {
                if (here->VDMOSsNodePrime == 0) {
                error = CKTmkVolt(ckt,&tmp,here->VDMOSname,"source");
                if(error) return(error);
                here->VDMOSsNodePrime = tmp->number;
                
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
                here->VDMOSsNodePrime = here->VDMOSsNode;
            }

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)
            TSTALLOC(VDMOSDdPtr,VDMOSdNode,VDMOSdNode);
            TSTALLOC(VDMOSGgPtr,VDMOSgNode,VDMOSgNode);
            TSTALLOC(VDMOSSsPtr,VDMOSsNode,VDMOSsNode);
            TSTALLOC(VDMOSBbPtr,VDMOSbNode,VDMOSbNode);
            TSTALLOC(VDMOSDPdpPtr,VDMOSdNodePrime,VDMOSdNodePrime);
            TSTALLOC(VDMOSSPspPtr,VDMOSsNodePrime,VDMOSsNodePrime);
            TSTALLOC(VDMOSDdpPtr,VDMOSdNode,VDMOSdNodePrime);
            TSTALLOC(VDMOSGbPtr,VDMOSgNode,VDMOSbNode);
            TSTALLOC(VDMOSGdpPtr,VDMOSgNode,VDMOSdNodePrime);
            TSTALLOC(VDMOSGspPtr,VDMOSgNode,VDMOSsNodePrime);
            TSTALLOC(VDMOSSspPtr,VDMOSsNode,VDMOSsNodePrime);
            TSTALLOC(VDMOSBdpPtr,VDMOSbNode,VDMOSdNodePrime);
            TSTALLOC(VDMOSBspPtr,VDMOSbNode,VDMOSsNodePrime);
            TSTALLOC(VDMOSDPspPtr,VDMOSdNodePrime,VDMOSsNodePrime);
            TSTALLOC(VDMOSDPdPtr,VDMOSdNodePrime,VDMOSdNode);
            TSTALLOC(VDMOSBgPtr,VDMOSbNode,VDMOSgNode);
            TSTALLOC(VDMOSDPgPtr,VDMOSdNodePrime,VDMOSgNode);
            TSTALLOC(VDMOSSPgPtr,VDMOSsNodePrime,VDMOSgNode);
            TSTALLOC(VDMOSSPsPtr,VDMOSsNodePrime,VDMOSsNode);
            TSTALLOC(VDMOSDPbPtr,VDMOSdNodePrime,VDMOSbNode);
            TSTALLOC(VDMOSSPbPtr,VDMOSsNodePrime,VDMOSbNode);
            TSTALLOC(VDMOSSPdpPtr,VDMOSsNodePrime,VDMOSdNodePrime);

        }
    }
    return(OK);
}

int
VDMOSunsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    VDMOSmodel *model;
    VDMOSinstance *here;

    for (model = (VDMOSmodel *)inModel; model != NULL;
	    model = VDMOSnextModel(model))
    {
        for (here = VDMOSinstances(model); here != NULL;
                here=VDMOSnextInstance(here))
	{
	    if (here->VDMOSsNodePrime > 0
		    && here->VDMOSsNodePrime != here->VDMOSsNode)
		CKTdltNNum(ckt, here->VDMOSsNodePrime);
            here->VDMOSsNodePrime= 0;

	    if (here->VDMOSdNodePrime > 0
		    && here->VDMOSdNodePrime != here->VDMOSdNode)
		CKTdltNNum(ckt, here->VDMOSdNodePrime);
            here->VDMOSdNodePrime= 0;
	}
    }
    return OK;
}

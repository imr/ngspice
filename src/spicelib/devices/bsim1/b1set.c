/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Hong J. Park, Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "bsim1def.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
B1setup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, 
        int *states)
        /* load the B1 device structure with those pointers needed later 
         * for fast matrix loading 
         */

{
    B1model *model = (B1model*)inModel;
    B1instance *here;
    int error;
    CKTnode *tmp;

    /*  loop through all the B1 device models */
    for( ; model != NULL; model = B1nextModel(model)) {
    
/* Default value Processing for B1 MOSFET Models */
        if( ! model->B1typeGiven) {
            model->B1type = NMOS;  /* NMOS */
        }
        if( ! model->B1vfb0Given) {
            model->B1vfb0 = 0.0;
        }
        if( ! model->B1vfbLGiven) {
            model->B1vfbL = 0.0;
        }
        if( ! model->B1vfbWGiven) {
            model->B1vfbW = 0.0;
        }
        if( ! model->B1phi0Given) {
            model->B1phi0 = 0.0;
        }
        if( ! model->B1phiLGiven) {
            model->B1phiL = 0.0;
        }
        if( ! model->B1phiWGiven) {
            model->B1phiW = 0.0;
        }
        if( ! model->B1K10Given) {
            model->B1K10 = 0.0;
        }
        if( ! model->B1K1LGiven) {
            model->B1K1L = 0.0;
        }
        if( ! model->B1K1WGiven) {
            model->B1K1W = 0.0;
        }
        if( ! model->B1K20Given) {
            model->B1K20 = 0.0;
        }
        if( ! model->B1K2LGiven) {
            model->B1K2L = 0.0;
        }
        if( ! model->B1K2WGiven) {
            model->B1K2W = 0.0;
        }
        if( ! model->B1eta0Given) {
            model->B1eta0 = 0.0;
        }
        if( ! model->B1etaLGiven) {
            model->B1etaL = 0.0;
        }
        if( ! model->B1etaWGiven) {
            model->B1etaW = 0.0;
        }
        if( ! model->B1mobZeroGiven) {
            model->B1mobZero = 0.0;
        }
        if( ! model->B1deltaLGiven) {
            model->B1deltaL = 0.0;
        }
        if( ! model->B1deltaWGiven) {
            model->B1deltaW = 0.0;
        }
        if( ! model->B1ugs0Given) {
            model->B1ugs0 = 0.0;
        }
        if( ! model->B1ugsLGiven) {
            model->B1ugsL = 0.0;
        }
        if( ! model->B1ugsWGiven) {
            model->B1ugsW = 0.0;
        }
        if( ! model->B1uds0Given) {
            model->B1uds0 = 0.0;
        }
        if( ! model->B1udsLGiven) {
            model->B1udsL = 0.0;
        }
        if( ! model->B1udsWGiven) {
            model->B1udsW = 0.0;
        }
        if( ! model->B1mobZeroB0Given) {
            model->B1mobZeroB0 = 0.0;
        }
        if( ! model->B1mobZeroBlGiven) {
            model->B1mobZeroBl = 0.0;
        }
        if( ! model->B1mobZeroBwGiven) {
            model->B1mobZeroBw = 0.0;
        }
        if( ! model->B1etaB0Given) {
            model->B1etaB0 = 0.0;
        }
        if( ! model->B1etaBlGiven) {
            model->B1etaBl = 0.0;
        }
        if( ! model->B1etaBwGiven) {
            model->B1etaBw = 0.0;
        }
        if( ! model->B1etaD0Given) {
            model->B1etaD0 = 0.0;
        }
        if( ! model->B1etaDlGiven) {
            model->B1etaDl = 0.0;
        }
        if( ! model->B1etaDwGiven) {
            model->B1etaDw = 0.0;
        }
        if( ! model->B1ugsB0Given) {
            model->B1ugsB0 = 0.0;
        }
        if( ! model->B1ugsBLGiven) {
            model->B1ugsBL = 0.0;
        }
        if( ! model->B1ugsBWGiven) {
            model->B1ugsBW = 0.0;
        }
        if( ! model->B1udsB0Given) {
            model->B1udsB0 = 0.0;
        }
        if( ! model->B1udsBLGiven) {
            model->B1udsBL = 0.0;
        }
        if( ! model->B1udsBWGiven) {
            model->B1udsBW = 0.0;
        }
        if( ! model->B1mobVdd0Given) {
            model->B1mobVdd0 = 0.0;
        }
        if( ! model->B1mobVddlGiven) {
            model->B1mobVddl = 0.0;
        }
        if( ! model->B1mobVddwGiven) {
            model->B1mobVddw = 0.0;
        }
        if( ! model->B1mobVddB0Given) {
            model->B1mobVddB0 = 0.0;
        }
        if( ! model->B1mobVddBlGiven) {
            model->B1mobVddBl = 0.0;
        }
        if( ! model->B1mobVddBwGiven) {
            model->B1mobVddBw = 0.0;
        }
        if( ! model->B1mobVddD0Given) {
            model->B1mobVddD0 = 0.0;
        }
        if( ! model->B1mobVddDlGiven) {
            model->B1mobVddDl = 0.0;
        }
        if( ! model->B1mobVddDwGiven) {
            model->B1mobVddDw = 0.0;
        }
        if( ! model->B1udsD0Given) {
            model->B1udsD0 = 0.0;
        }
        if( ! model->B1udsDLGiven) {
            model->B1udsDL = 0.0;
        }
        if( ! model->B1udsDWGiven) {
            model->B1udsDW = 0.0;
        }
        if( ! model->B1oxideThicknessGiven) {
            model->B1oxideThickness = 0.0;  /* um */
        }
        if( ! model->B1tempGiven) {
            model->B1temp = 0.0;
        }
        if( ! model->B1vddGiven) {
            model->B1vdd = 0.0;
        }
        if( ! model->B1gateDrainOverlapCapGiven) {
            model->B1gateDrainOverlapCap = 0.0;
        }
        if( ! model->B1gateSourceOverlapCapGiven) {
            model->B1gateSourceOverlapCap = 0.0;
        }
        if( ! model->B1gateBulkOverlapCapGiven) {
            model->B1gateBulkOverlapCap = 0.0;
        }
        if( ! model->B1channelChargePartitionFlagGiven) {
            model->B1channelChargePartitionFlag = 0;
        }
        if( ! model->B1subthSlope0Given) {
            model->B1subthSlope0 = 0.0;
        }
        if( ! model->B1subthSlopeLGiven) {
            model->B1subthSlopeL = 0.0;
        }
        if( ! model->B1subthSlopeWGiven) {
            model->B1subthSlopeW = 0.0;
        }
        if( ! model->B1subthSlopeB0Given) {
            model->B1subthSlopeB0 = 0.0;
        }
        if( ! model->B1subthSlopeBLGiven) {
            model->B1subthSlopeBL = 0.0;
        }
        if( ! model->B1subthSlopeBWGiven) {
            model->B1subthSlopeBW = 0.0;
        }
        if( ! model->B1subthSlopeD0Given) {
            model->B1subthSlopeD0 = 0.0;
        }
        if( ! model->B1subthSlopeDLGiven) {
            model->B1subthSlopeDL = 0.0;
        }
        if( ! model->B1subthSlopeDWGiven) {
            model->B1subthSlopeDW = 0.0;
        }
        if( ! model->B1sheetResistanceGiven) {
            model->B1sheetResistance = 0.0;
        }
        if( ! model->B1unitAreaJctCapGiven) {
            model->B1unitAreaJctCap = 0.0;
        }
        if( ! model->B1unitLengthSidewallJctCapGiven) {
            model->B1unitLengthSidewallJctCap = 0.0;
        }
        if( ! model->B1jctSatCurDensityGiven) {
            model->B1jctSatCurDensity = 0.0;
        }
        if( ! model->B1bulkJctPotentialGiven) {
            model->B1bulkJctPotential = 0.0;
        }
        if( ! model->B1sidewallJctPotentialGiven) {
            model->B1sidewallJctPotential = 0.0;
        }
        if( ! model->B1bulkJctBotGradingCoeffGiven) {
            model->B1bulkJctBotGradingCoeff = 0.0;
        }
        if( ! model->B1bulkJctSideGradingCoeffGiven) {
            model->B1bulkJctSideGradingCoeff = 0.0;
        }
        if( ! model->B1defaultWidthGiven) {
            model->B1defaultWidth = 0.0;
        }
        if( ! model->B1deltaLengthGiven) {
            model->B1deltaLength = 0.0;
        }
        if( ! model->B1fNcoefGiven) {
            model->B1fNcoef = 0.0;
        }
        if( ! model->B1fNexpGiven) {
            model->B1fNexp = 1.0;
        }

        /* loop through all the instances of the model */
        for (here = B1instances(model); here != NULL ;
                here=B1nextInstance(here)) {

        CKTnode *tmpNode;
        IFuid tmpName;

            /* allocate a chunk of the state vector */
            here->B1states = *states;
            *states += B1numStates;

            /* perform the parameter defaulting */
            if(!here->B1drainAreaGiven) {
                here->B1drainArea = 0;
            }
            if(!here->B1drainPerimeterGiven) {
                here->B1drainPerimeter = 0;
            }
            if(!here->B1drainSquaresGiven) {
                here->B1drainSquares = 1;
            }
            if(!here->B1icVBSGiven) {
                here->B1icVBS = 0;
            }
            if(!here->B1icVDSGiven) {
                here->B1icVDS = 0;
            }
            if(!here->B1icVGSGiven) {
                here->B1icVGS = 0;
            }
            if(!here->B1lGiven) {
                here->B1l = 5e-6;
            }
            if(!here->B1sourceAreaGiven) {
                here->B1sourceArea = 0;
            }
            if(!here->B1sourcePerimeterGiven) {
                here->B1sourcePerimeter = 0;
            }
            if(!here->B1sourceSquaresGiven) {
                here->B1sourceSquares = 1;
            }
            if(!here->B1vdsatGiven) {
                here->B1vdsat = 0;
            }
            if(!here->B1vonGiven) {
                here->B1von = 0;
            }
            if(!here->B1wGiven) {
                here->B1w = 5e-6;
            }
            if(!here->B1mGiven) {
                here->B1m = 1.0;
            }
            
            /* process drain series resistance */
            if( (model->B1sheetResistance != 0) && 
                    (here->B1drainSquares != 0.0 ))
            {   if(here->B1dNodePrime == 0) {
                error = CKTmkVolt(ckt,&tmp,here->B1name,"drain");
                if(error) return(error);
                here->B1dNodePrime = tmp->number;
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
                    here->B1dNodePrime = here->B1dNode;
            }
                   
            /* process source series resistance */
            if( (model->B1sheetResistance != 0) && 
                    (here->B1sourceSquares != 0.0 )) {
                if(here->B1sNodePrime == 0) {
                    error = CKTmkVolt(ckt,&tmp,here->B1name,"source");
                    if(error) return(error);
                    here->B1sNodePrime = tmp->number;
                    if (ckt->CKTcopyNodesets) {
                     if (CKTinst2Node(ckt,here,3,&tmpNode,&tmpName)==OK) {
                     if (tmpNode->nsGiven) {
                       tmp->nodeset=tmpNode->nodeset; 
                       tmp->nsGiven=tmpNode->nsGiven; 
                     }
                  }
                }
                }
            } else  {
                here->B1sNodePrime = here->B1sNode;
            }
                   
        /* set Sparse Matrix Pointers */

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

            TSTALLOC(B1DdPtr, B1dNode, B1dNode);
            TSTALLOC(B1GgPtr, B1gNode, B1gNode);
            TSTALLOC(B1SsPtr, B1sNode, B1sNode);
            TSTALLOC(B1BbPtr, B1bNode, B1bNode);
            TSTALLOC(B1DPdpPtr, B1dNodePrime, B1dNodePrime);
            TSTALLOC(B1SPspPtr, B1sNodePrime, B1sNodePrime);
            TSTALLOC(B1DdpPtr, B1dNode, B1dNodePrime);
            TSTALLOC(B1GbPtr, B1gNode, B1bNode);
            TSTALLOC(B1GdpPtr, B1gNode, B1dNodePrime);
            TSTALLOC(B1GspPtr, B1gNode, B1sNodePrime);
            TSTALLOC(B1SspPtr, B1sNode, B1sNodePrime);
            TSTALLOC(B1BdpPtr, B1bNode, B1dNodePrime);
            TSTALLOC(B1BspPtr, B1bNode, B1sNodePrime);
            TSTALLOC(B1DPspPtr, B1dNodePrime, B1sNodePrime);
            TSTALLOC(B1DPdPtr, B1dNodePrime, B1dNode);
            TSTALLOC(B1BgPtr, B1bNode, B1gNode);
            TSTALLOC(B1DPgPtr, B1dNodePrime, B1gNode);
            TSTALLOC(B1SPgPtr, B1sNodePrime, B1gNode);
            TSTALLOC(B1SPsPtr, B1sNodePrime, B1sNode);
            TSTALLOC(B1DPbPtr, B1dNodePrime, B1bNode);
            TSTALLOC(B1SPbPtr, B1sNodePrime, B1bNode);
            TSTALLOC(B1SPdpPtr, B1sNodePrime, B1dNodePrime);

        }
    }
    return(OK);
}  

int
B1unsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    B1model *model;
    B1instance *here;

    for (model = (B1model *)inModel; model != NULL;
	    model = B1nextModel(model))
    {
        for (here = B1instances(model); here != NULL;
                here=B1nextInstance(here))
	{
	    if (here->B1sNodePrime > 0
		    && here->B1sNodePrime != here->B1sNode)
		CKTdltNNum(ckt, here->B1sNodePrime);
            here->B1sNodePrime = 0;

	    if (here->B1dNodePrime > 0
		    && here->B1dNodePrime != here->B1dNode)
		CKTdltNNum(ckt, here->B1dNodePrime);
            here->B1dNodePrime = 0;
	}
    }
    return OK;
}

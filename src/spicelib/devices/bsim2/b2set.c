/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Min-Chie Jeng, Hong J. Park, Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "bsim2def.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
B2setup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
        /* load the B2 device structure with those pointers needed later 
         * for fast matrix loading 
         */

{
    B2model *model = (B2model*)inModel;
    B2instance *here;
    int error;
    CKTnode *tmp;

    /*  loop through all the B2 device models */
    for( ; model != NULL; model = B2nextModel(model)) {
    
/* Default value Processing for B2 MOSFET Models */
        if( ! model->B2typeGiven) {
            model->B2type = NMOS;  /* NMOS */
        }
        if( ! model->B2vfb0Given) {
            model->B2vfb0 = -1.0;
        }
        if( ! model->B2vfbLGiven) {
            model->B2vfbL = 0.0;
        }
        if( ! model->B2vfbWGiven) {
            model->B2vfbW = 0.0;
        }
        if( ! model->B2phi0Given) {
            model->B2phi0 = 0.75;
        }
        if( ! model->B2phiLGiven) {
            model->B2phiL = 0.0;
        }
        if( ! model->B2phiWGiven) {
            model->B2phiW = 0.0;
        }
        if( ! model->B2k10Given) {
            model->B2k10 = 0.8;
        }
        if( ! model->B2k1LGiven) {
            model->B2k1L = 0.0;
        }
        if( ! model->B2k1WGiven) {
            model->B2k1W = 0.0;
        }
        if( ! model->B2k20Given) {
            model->B2k20 = 0.0;
        }
        if( ! model->B2k2LGiven) {
            model->B2k2L = 0.0;
        }
        if( ! model->B2k2WGiven) {
            model->B2k2W = 0.0;
        }
        if( ! model->B2eta00Given) {
            model->B2eta00 = 0.0;
        }
        if( ! model->B2eta0LGiven) {
            model->B2eta0L = 0.0;
        }
        if( ! model->B2eta0WGiven) {
            model->B2eta0W = 0.0;
        }
        if( ! model->B2etaB0Given) {
            model->B2etaB0 = 0.0;
        }
        if( ! model->B2etaBLGiven) {
            model->B2etaBL = 0.0;
        }
        if( ! model->B2etaBWGiven) {
            model->B2etaBW = 0.0;
        }
        if( ! model->B2deltaLGiven) {
            model->B2deltaL = 0.0;
        }
        if( ! model->B2deltaWGiven) {
            model->B2deltaW = 0.0;
        }
        if( ! model->B2ua00Given) {
            model->B2ua00 = 0.2;
        }
        if( ! model->B2ua0LGiven) {
            model->B2ua0L = 0.0;
        }
        if( ! model->B2ua0WGiven) {
            model->B2ua0W = 0.0;
        }
        if( ! model->B2uaB0Given) {
            model->B2uaB0 = 0.0;
        }
        if( ! model->B2uaBLGiven) {
            model->B2uaBL = 0.0;
        }
        if( ! model->B2uaBWGiven) {
            model->B2uaBW = 0.0;
        }
        if( ! model->B2ub00Given) {
            model->B2ub00 = 0.0;
        }
        if( ! model->B2ub0LGiven) {
            model->B2ub0L = 0.0;
        }
        if( ! model->B2ub0WGiven) {
            model->B2ub0W = 0.0;
        }
        if( ! model->B2ubB0Given) {
            model->B2ubB0 = 0.0;
        }
        if( ! model->B2ubBLGiven) {
            model->B2ubBL = 0.0;
        }
        if( ! model->B2ubBWGiven) {
            model->B2ubBW = 0.0;
        }
        if( ! model->B2u100Given) {
            model->B2u100 = 0.1;
        }
        if( ! model->B2u10LGiven) {
            model->B2u10L = 0.0;
        }
        if( ! model->B2u10WGiven) {
            model->B2u10W = 0.0;
        }
        if( ! model->B2u1B0Given) {
            model->B2u1B0 = 0.0;
        }
        if( ! model->B2u1BLGiven) {
            model->B2u1BL = 0.0;
        }
        if( ! model->B2u1BWGiven) {
            model->B2u1BW = 0.0;
        }
        if( ! model->B2u1D0Given) {
            model->B2u1D0 = 0.0;
        }
        if( ! model->B2u1DLGiven) {
            model->B2u1DL = 0.0;
        }
        if( ! model->B2u1DWGiven) {
            model->B2u1DW = 0.0;
        }
        if( ! model->B2mob00Given) {
            model->B2mob00 = 400.0;
        }
        if( ! model->B2mob0B0Given) {
            model->B2mob0B0 = 0.0;
        }
        if( ! model->B2mob0BLGiven) {
            model->B2mob0BL = 0.0;
        }
        if( ! model->B2mob0BWGiven) {
            model->B2mob0BW = 0.0;
        }
        if( ! model->B2mobs00Given) {
            model->B2mobs00 = 500.0;
        }
        if( ! model->B2mobs0LGiven) {
            model->B2mobs0L = 0.0;
        }
        if( ! model->B2mobs0WGiven) {
            model->B2mobs0W = 0.0;
        }
        if( ! model->B2mobsB0Given) {
            model->B2mobsB0 = 0.0;
        }
        if( ! model->B2mobsBLGiven) {
            model->B2mobsBL = 0.0;
        }
        if( ! model->B2mobsBWGiven) {
            model->B2mobsBW = 0.0;
        }
        if( ! model->B2mob200Given) {
            model->B2mob200 = 1.5;
        }
        if( ! model->B2mob20LGiven) {
            model->B2mob20L = 0.0;
        }
        if( ! model->B2mob20WGiven) {
            model->B2mob20W = 0.0;
        }
        if( ! model->B2mob2B0Given) {
            model->B2mob2B0 = 0.0;
        }
        if( ! model->B2mob2BLGiven) {
            model->B2mob2BL = 0.0;
        }
        if( ! model->B2mob2BWGiven) {
            model->B2mob2BW = 0.0;
        }
        if( ! model->B2mob2G0Given) {
            model->B2mob2G0 = 0.0;
        }
        if( ! model->B2mob2GLGiven) {
            model->B2mob2GL = 0.0;
        }
        if( ! model->B2mob2GWGiven) {
            model->B2mob2GW = 0.0;
        }
        if( ! model->B2mob300Given) {
            model->B2mob300 = 10;
        }
        if( ! model->B2mob30LGiven) {
            model->B2mob30L = 0.0;
        }
        if( ! model->B2mob30WGiven) {
            model->B2mob30W = 0.0;
        }
        if( ! model->B2mob3B0Given) {
            model->B2mob3B0 = 0.0;
        }
        if( ! model->B2mob3BLGiven) {
            model->B2mob3BL = 0.0;
        }
        if( ! model->B2mob3BWGiven) {
            model->B2mob3BW = 0.0;
        }
        if( ! model->B2mob3G0Given) {
            model->B2mob3G0 = 0.0;
        }
        if( ! model->B2mob3GLGiven) {
            model->B2mob3GL = 0.0;
        }
        if( ! model->B2mob3GWGiven) {
            model->B2mob3GW = 0.0;
        }
        if( ! model->B2mob400Given) {
            model->B2mob400 = 0.0;
        }
        if( ! model->B2mob40LGiven) {
            model->B2mob40L = 0.0;
        }
        if( ! model->B2mob40WGiven) {
            model->B2mob40W = 0.0;
        }
        if( ! model->B2mob4B0Given) {
            model->B2mob4B0 = 0.0;
        }
        if( ! model->B2mob4BLGiven) {
            model->B2mob4BL = 0.0;
        }
        if( ! model->B2mob4BWGiven) {
            model->B2mob4BW = 0.0;
        }
        if( ! model->B2mob4G0Given) {
            model->B2mob4G0 = 0.0;
        }
        if( ! model->B2mob4GLGiven) {
            model->B2mob4GL = 0.0;
        }
        if( ! model->B2mob4GWGiven) {
            model->B2mob4GW = 0.0;
        }
        if( ! model->B2n00Given) {
            model->B2n00 = 1.4;
        }
        if( ! model->B2n0LGiven) {
            model->B2n0L = 0.0;
        }
        if( ! model->B2n0WGiven) {
            model->B2n0W = 0.0;
        }
        if( ! model->B2nB0Given) {
            model->B2nB0 = 0.5;
        }
        if( ! model->B2nBLGiven) {
            model->B2nBL = 0.0;
        }
        if( ! model->B2nBWGiven) {
            model->B2nBW = 0.0;
        }
        if( ! model->B2nD0Given) {
            model->B2nD0 = 0.0;
        }
        if( ! model->B2nDLGiven) {
            model->B2nDL = 0.0;
        }
        if( ! model->B2nDWGiven) {
            model->B2nDW = 0.0;
        }
        if( ! model->B2vof00Given) {
            model->B2vof00 = 1.8;
        }
        if( ! model->B2vof0LGiven) {
            model->B2vof0L = 0.0;
        }
        if( ! model->B2vof0WGiven) {
            model->B2vof0W = 0.0;
        }
        if( ! model->B2vofB0Given) {
            model->B2vofB0 = 0.0;
        }
        if( ! model->B2vofBLGiven) {
            model->B2vofBL = 0.0;
        }
        if( ! model->B2vofBWGiven) {
            model->B2vofBW = 0.0;
        }
        if( ! model->B2vofD0Given) {
            model->B2vofD0 = 0.0;
        }
        if( ! model->B2vofDLGiven) {
            model->B2vofDL = 0.0;
        }
        if( ! model->B2vofDWGiven) {
            model->B2vofDW = 0.0;
        }
        if( ! model->B2ai00Given) {
            model->B2ai00 = 0.0;
        }
        if( ! model->B2ai0LGiven) {
            model->B2ai0L = 0.0;
        }
        if( ! model->B2ai0WGiven) {
            model->B2ai0W = 0.0;
        }
        if( ! model->B2aiB0Given) {
            model->B2aiB0 = 0.0;
        }
        if( ! model->B2aiBLGiven) {
            model->B2aiBL = 0.0;
        }
        if( ! model->B2aiBWGiven) {
            model->B2aiBW = 0.0;
        }
        if( ! model->B2bi00Given) {
            model->B2bi00 = 0.0;
        }
        if( ! model->B2bi0LGiven) {
            model->B2bi0L = 0.0;
        }
        if( ! model->B2bi0WGiven) {
            model->B2bi0W = 0.0;
        }
        if( ! model->B2biB0Given) {
            model->B2biB0 = 0.0;
        }
        if( ! model->B2biBLGiven) {
            model->B2biBL = 0.0;
        }
        if( ! model->B2biBWGiven) {
            model->B2biBW = 0.0;
        }
        if( ! model->B2vghigh0Given) {
            model->B2vghigh0 = 0.2;
        }
        if( ! model->B2vghighLGiven) {
            model->B2vghighL = 0.0;
        }
        if( ! model->B2vghighWGiven) {
            model->B2vghighW = 0.0;
        }
        if( ! model->B2vglow0Given) {
            model->B2vglow0 = -0.15;
        }
        if( ! model->B2vglowLGiven) {
            model->B2vglowL = 0.0;
        }
        if( ! model->B2vglowWGiven) {
            model->B2vglowW = 0.0;
        }
        if( ! model->B2toxGiven) {
            model->B2tox = 0.03;  /* um */
        }
        if( ! model->B2tempGiven) {
            model->B2temp = 27.0;
        }
        if( ! model->B2vddGiven) {
            model->B2vdd = 5.0;
        }
        if( ! model->B2vggGiven) {
            model->B2vgg = 5.0;
        }
        if( ! model->B2vbbGiven) {
            model->B2vbb = 5.0;
        }
        if( ! model->B2gateDrainOverlapCapGiven) {
            model->B2gateDrainOverlapCap = 0.0;
        }
        if( ! model->B2gateSourceOverlapCapGiven) {
            model->B2gateSourceOverlapCap = 0.0;
        }
        if( ! model->B2gateBulkOverlapCapGiven) {
            model->B2gateBulkOverlapCap = 0.0;
        }
        if( ! model->B2channelChargePartitionFlagGiven) {
            model->B2channelChargePartitionFlag = 0;
        }
        if( ! model->B2sheetResistanceGiven) {
            model->B2sheetResistance = 0.0;
        }
        if( ! model->B2unitAreaJctCapGiven) {
            model->B2unitAreaJctCap = 0.0;
        }
        if( ! model->B2unitLengthSidewallJctCapGiven) {
            model->B2unitLengthSidewallJctCap = 0.0;
        }
        if( ! model->B2jctSatCurDensityGiven) {
            model->B2jctSatCurDensity = 0.0;
        }
        if( ! model->B2bulkJctPotentialGiven) {
            model->B2bulkJctPotential = 0.0;
        }
        if( ! model->B2sidewallJctPotentialGiven) {
            model->B2sidewallJctPotential = 0.0;
        }
        if( ! model->B2bulkJctBotGradingCoeffGiven) {
            model->B2bulkJctBotGradingCoeff = 0.0;
        }
        if( ! model->B2bulkJctSideGradingCoeffGiven) {
            model->B2bulkJctSideGradingCoeff = 0.0;
        }
        if( ! model->B2defaultWidthGiven) {
            model->B2defaultWidth = 10.0;
        }
        if( ! model->B2deltaLengthGiven) {
            model->B2deltaLength = 0.0;
        }

        /* loop through all the instances of the model */
        for (here = B2instances(model); here != NULL ;
                here=B2nextInstance(here)) {

            /* allocate a chunk of the state vector */
            here->B2states = *states;
            *states += B2numStates;

            /* perform the parameter defaulting */

            if(!here->B2drainAreaGiven) {
                here->B2drainArea = 0;
            }
            if(!here->B2drainPerimeterGiven) {
                here->B2drainPerimeter = 0;
            }
            if(!here->B2drainSquaresGiven) {
                here->B2drainSquares = 1;
            }
            if(!here->B2icVBSGiven) {
                here->B2icVBS = 0;
            }
            if(!here->B2icVDSGiven) {
                here->B2icVDS = 0;
            }
            if(!here->B2icVGSGiven) {
                here->B2icVGS = 0;
            }
            if(!here->B2lGiven) {
                here->B2l = 5e-6;
            }
            if(!here->B2sourceAreaGiven) {
                here->B2sourceArea = 0;
            }
            if(!here->B2sourcePerimeterGiven) {
                here->B2sourcePerimeter = 0;
            }
            if(!here->B2sourceSquaresGiven) {
                here->B2sourceSquares = 1;
            }
            if(!here->B2vdsatGiven) {
                here->B2vdsat = 0;
            }
            if(!here->B2vonGiven) {
                here->B2von = 0;
            }
            if(!here->B2wGiven) {
                here->B2w = 5e-6;
            }
            if(!here->B2mGiven) {
                here->B2m = 1.0;
            }
       
            /* process drain series resistance */
            if( (model->B2sheetResistance != 0) && 
                    (here->B2drainSquares != 0.0 ) ) {
               if (here->B2dNodePrime == 0) {
                error = CKTmkVolt(ckt,&tmp,here->B2name,"drain");
               if(error) return(error);
                here->B2dNodePrime = tmp->number;
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
                    here->B2dNodePrime = here->B2dNode;
            }
                   
            /* process source series resistance */
            if( (model->B2sheetResistance != 0) && 
                    (here->B2sourceSquares != 0.0 ) ) {
                if(here->B2sNodePrime == 0) {
                    error = CKTmkVolt(ckt,&tmp,here->B2name,"source");
                    if(error)
			return(error);
                    here->B2sNodePrime = tmp->number;
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
            } else  {
                here->B2sNodePrime = here->B2sNode;
            }
                   

        /* set Sparse Matrix Pointers */

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

            TSTALLOC(B2DdPtr, B2dNode, B2dNode);
            TSTALLOC(B2GgPtr, B2gNode, B2gNode);
            TSTALLOC(B2SsPtr, B2sNode, B2sNode);
            TSTALLOC(B2BbPtr, B2bNode, B2bNode);
            TSTALLOC(B2DPdpPtr, B2dNodePrime, B2dNodePrime);
            TSTALLOC(B2SPspPtr, B2sNodePrime, B2sNodePrime);
            TSTALLOC(B2DdpPtr, B2dNode, B2dNodePrime);
            TSTALLOC(B2GbPtr, B2gNode, B2bNode);
            TSTALLOC(B2GdpPtr, B2gNode, B2dNodePrime);
            TSTALLOC(B2GspPtr, B2gNode, B2sNodePrime);
            TSTALLOC(B2SspPtr, B2sNode, B2sNodePrime);
            TSTALLOC(B2BdpPtr, B2bNode, B2dNodePrime);
            TSTALLOC(B2BspPtr, B2bNode, B2sNodePrime);
            TSTALLOC(B2DPspPtr, B2dNodePrime, B2sNodePrime);
            TSTALLOC(B2DPdPtr, B2dNodePrime, B2dNode);
            TSTALLOC(B2BgPtr, B2bNode, B2gNode);
            TSTALLOC(B2DPgPtr, B2dNodePrime, B2gNode);
            TSTALLOC(B2SPgPtr, B2sNodePrime, B2gNode);
            TSTALLOC(B2SPsPtr, B2sNodePrime, B2sNode);
            TSTALLOC(B2DPbPtr, B2dNodePrime, B2bNode);
            TSTALLOC(B2SPbPtr, B2sNodePrime, B2bNode);
            TSTALLOC(B2SPdpPtr, B2sNodePrime, B2dNodePrime);

        }
    }
    return(OK);
}  

int
B2unsetup(
    GENmodel *inModel,
    CKTcircuit *ckt)
{
    B2model *model;
    B2instance *here;

    for (model = (B2model *)inModel; model != NULL;
	    model = B2nextModel(model))
    {
        for (here = B2instances(model); here != NULL;
                here=B2nextInstance(here))
	{
	    if (here->B2sNodePrime > 0
		    && here->B2sNodePrime != here->B2sNode)
		CKTdltNNum(ckt, here->B2sNodePrime);
            here->B2sNodePrime = 0;

	    if (here->B2dNodePrime > 0
		    && here->B2dNodePrime != here->B2dNode)
		CKTdltNNum(ckt, here->B2dNodePrime);
            here->B2dNodePrime = 0;
	}
    }
    return OK;
}

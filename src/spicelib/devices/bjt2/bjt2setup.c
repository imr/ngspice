/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/

/* 
 * This routine should only be called when circuit topology
 * changes, since its computations do not depend on most
 * device or model parameters, only on topology (as
 * affected by emitter, collector, and base resistances)
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "smpdefs.h"
#include "bjt2defs.h"
#include "const.h"
#include "sperror.h"
#include "ifsim.h"
#include "suffix.h"

int
BJT2setup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
        /* load the BJT2 structure with those pointers needed later 
         * for fast matrix loading 
         */

{
    BJT2model *model = (BJT2model*)inModel;
    BJT2instance *here;
    int error;
    CKTnode *tmp;

    /*  loop through all the diode models */
    for( ; model != NULL; model = model->BJT2nextModel ) {

        if(model->BJT2type != NPN && model->BJT2type != PNP) {
            model->BJT2type = NPN;
        }
#ifndef GEOMETRY_COMPAT
        if(!model->BJT2subsGiven ||
           (model->BJT2subs != VERTICAL && model->BJT2subs != LATERAL)) {
            model->BJT2subs = VERTICAL;
        }
#else
        if(!model->BJT2subsGiven ||
           (model->BJT2subs != VERTICAL && model->BJT2subs != LATERAL)) {
            if (model->BJT2type = NPN) 
	        model->BJT2subs = VERTICAL;  /* Vertical for NPN */
		else
		model->BJT2subs = LATERAL;   /* Lateral for PNP */
        }	
#endif
        if(!model->BJT2satCurGiven) {
            model->BJT2satCur = 1e-16;
        }
        if(!model->BJT2subSatCurGiven) {
            model->BJT2subSatCur = 1e-16;
        }
        if(!model->BJT2betaFGiven) {
            model->BJT2betaF = 100;
        }
        if(!model->BJT2emissionCoeffFGiven) {
            model->BJT2emissionCoeffF = 1;
        }
        if(!model->BJT2leakBEemissionCoeffGiven) {
            model->BJT2leakBEemissionCoeff = 1.5;
        }
        if(!model->BJT2betaRGiven) {
            model->BJT2betaR = 1;
        }
        if(!model->BJT2emissionCoeffRGiven) {
            model->BJT2emissionCoeffR = 1;
        }
        if(!model->BJT2leakBCemissionCoeffGiven) {
            model->BJT2leakBCemissionCoeff = 2;
        }
        if(!model->BJT2baseResistGiven) {
            model->BJT2baseResist = 0;
        }
        if(!model->BJT2emitterResistGiven) {
            model->BJT2emitterResist = 0;
        }
        if(!model->BJT2collectorResistGiven) {
            model->BJT2collectorResist = 0;
        }
        if(!model->BJT2depletionCapBEGiven) {
            model->BJT2depletionCapBE = 0;
        }
        if(!model->BJT2potentialBEGiven) {
            model->BJT2potentialBE = .75;
        }
        if(!model->BJT2junctionExpBEGiven) {
            model->BJT2junctionExpBE = .33;
        }
        if(!model->BJT2transitTimeFGiven) {
            model->BJT2transitTimeF = 0;
        }
        if(!model->BJT2transitTimeBiasCoeffFGiven) {
            model->BJT2transitTimeBiasCoeffF = 0;
        }
        if(!model->BJT2transitTimeHighCurrentFGiven) {
            model->BJT2transitTimeHighCurrentF = 0;
        }
        if(!model->BJT2excessPhaseGiven) {
            model->BJT2excessPhase = 0;
        }
        if(!model->BJT2depletionCapBCGiven) {
            model->BJT2depletionCapBC = 0;
        }
        if(!model->BJT2potentialBCGiven) {
            model->BJT2potentialBC = .75;
        }
        if(!model->BJT2junctionExpBCGiven) {
            model->BJT2junctionExpBC = .33;
        }
        if(!model->BJT2baseFractionBCcapGiven) {
            model->BJT2baseFractionBCcap = 1;
        }
        if(!model->BJT2transitTimeRGiven) {
            model->BJT2transitTimeR = 0;
        }
        if(!model->BJT2capSubGiven) {
            model->BJT2capSub = 0;
        }
        if(!model->BJT2potentialSubstrateGiven) {
            model->BJT2potentialSubstrate = .75;
        }
        if(!model->BJT2exponentialSubstrateGiven) {
            model->BJT2exponentialSubstrate = 0;
        }
        if(!model->BJT2betaExpGiven) {
            model->BJT2betaExp = 0;
        }
        if(!model->BJT2energyGapGiven) {
            model->BJT2energyGap = 1.11;
        }
        if(!model->BJT2tempExpISGiven) {
            model->BJT2tempExpIS = 3;
        }
        if(!model->BJT2reTempCoeff1Given) {
            model->BJT2reTempCoeff1 = 0.0;
        }
        if(!model->BJT2reTempCoeff2Given) {
            model->BJT2reTempCoeff2 = 0.0;
        }
        if(!model->BJT2rcTempCoeff1Given) {
            model->BJT2rcTempCoeff1 = 0.0;
        }
        if(!model->BJT2rcTempCoeff2Given) {
            model->BJT2rcTempCoeff2 = 0.0;
        }
        if(!model->BJT2rbTempCoeff1Given) {
            model->BJT2rbTempCoeff1 = 0.0;
        }
        if(!model->BJT2rbTempCoeff2Given) {
            model->BJT2rbTempCoeff2 = 0.0;
        }
        if(!model->BJT2rbmTempCoeff1Given) {
            model->BJT2rbmTempCoeff1 = 0.0;
        }
        if(!model->BJT2rbmTempCoeff2Given) {
            model->BJT2rbmTempCoeff2 = 0.0;
        }
	if(!model->BJT2fNcoefGiven) {
	    model->BJT2fNcoef = 0;
	}
	if(!model->BJT2fNexpGiven) {
	    model->BJT2fNexp = 1;
	}

/*
 * COMPATABILITY WARNING!
 * special note:  for backward compatability to much older models, spice 2G
 * implemented a special case which checked if B-E leakage saturation
 * current was >1, then it was instead a the B-E leakage saturation current
 * divided by IS, and multiplied it by IS at this point.  This was not
 * handled correctly in the 2G code, and there is some question on its 
 * reasonability, since it is also undocumented, so it has been left out
 * here.  It could easily be added with 1 line.  (The same applies to the B-C
 * leakage saturation current).   TQ  6/29/84
 */
            
        /* loop through all the instances of the model */
        for (here = model->BJT2instances; here != NULL ;
                here=here->BJT2nextInstance) {
            CKTnode *tmpNode;
            IFuid tmpName;

            if (here->BJT2owner != ARCHme)
                goto matrixpointers;


            if(!here->BJT2areaGiven) {
                here->BJT2area = 1;
            }
            if(!here->BJT2areabGiven) {
                here->BJT2areab = here->BJT2area;
            }
            if(!here->BJT2areacGiven) {
                here->BJT2areac = here->BJT2area;
            }

	    if(!here->BJT2mGiven) {
                here->BJT2m = 1.0;
            }
	    	
	    here->BJT2state = *states;
            *states += BJT2numStates;
            if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN) ){
                *states += 8 * (ckt->CKTsenInfo->SENparms);
            }	

matrixpointers:	        
            if(model->BJT2collectorResist == 0) {
                here->BJT2colPrimeNode = here->BJT2colNode;
            } else if(here->BJT2colPrimeNode == 0) {
                error = CKTmkVolt(ckt,&tmp,here->BJT2name,"collector");
                if(error) return(error);
                here->BJT2colPrimeNode = tmp->number;
                if (ckt->CKTcopyNodesets) {
                  if (CKTinst2Node(ckt,here,1,&tmpNode,&tmpName)==OK) {
                     if (tmpNode->nsGiven) {
                       tmp->nodeset=tmpNode->nodeset; 
                       tmp->nsGiven=tmpNode->nsGiven; 
/*                     fprintf(stderr, "Nodeset copied from %s\n", tmpName);
                       fprintf(stderr, "                 to %s\n", tmp->name);
                       fprintf(stderr, "              value %g\n",
                                                                tmp->nodeset);*/
                     }
                  }
                }
            }
            if(model->BJT2baseResist == 0) {
                here->BJT2basePrimeNode = here->BJT2baseNode;
            } else if(here->BJT2basePrimeNode == 0){
                error = CKTmkVolt(ckt,&tmp,here->BJT2name, "base");
                if(error) return(error);
                here->BJT2basePrimeNode = tmp->number;
                if (ckt->CKTcopyNodesets) {
                  if (CKTinst2Node(ckt,here,2,&tmpNode,&tmpName)==OK) {
                     if (tmpNode->nsGiven) {
                       tmp->nodeset=tmpNode->nodeset; 
                       tmp->nsGiven=tmpNode->nsGiven; 
/*                     fprintf(stderr, "Nodeset copied from %s\n", tmpName);
                       fprintf(stderr, "                 to %s\n", tmp->name);
                       fprintf(stderr, "              value %g\n",
                                                                tmp->nodeset);*/
                     }
                  }
                }
            }
            if(model->BJT2emitterResist == 0) {
                here->BJT2emitPrimeNode = here->BJT2emitNode;
            } else if(here->BJT2emitPrimeNode == 0) {
                error = CKTmkVolt(ckt,&tmp,here->BJT2name, "emitter");
                if(error) return(error);
                here->BJT2emitPrimeNode = tmp->number;
                if (ckt->CKTcopyNodesets) {
                  if (CKTinst2Node(ckt,here,3,&tmpNode,&tmpName)==OK) {
                     if (tmpNode->nsGiven) {
                       tmp->nodeset=tmpNode->nodeset; 
                       tmp->nsGiven=tmpNode->nsGiven; 
/*                     fprintf(stderr, "Nodeset copied from %s\n", tmpName);
                       fprintf(stderr, "                 to %s\n", tmp->name);
                       fprintf(stderr, "              value %g\n",
                                                                tmp->nodeset);*/
                     }
                  }
                }
            }

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
if((here->ptr = SMPmakeElt(matrix,here->first,here->second))==(double *)NULL){\
    return(E_NOMEM);\
}
            TSTALLOC(BJT2colColPrimePtr,BJT2colNode,BJT2colPrimeNode)
            TSTALLOC(BJT2baseBasePrimePtr,BJT2baseNode,BJT2basePrimeNode)
            TSTALLOC(BJT2emitEmitPrimePtr,BJT2emitNode,BJT2emitPrimeNode)
            TSTALLOC(BJT2colPrimeColPtr,BJT2colPrimeNode,BJT2colNode)
            TSTALLOC(BJT2colPrimeBasePrimePtr,BJT2colPrimeNode,BJT2basePrimeNode)
            TSTALLOC(BJT2colPrimeEmitPrimePtr,BJT2colPrimeNode,BJT2emitPrimeNode)
            TSTALLOC(BJT2basePrimeBasePtr,BJT2basePrimeNode,BJT2baseNode)
            TSTALLOC(BJT2basePrimeColPrimePtr,BJT2basePrimeNode,BJT2colPrimeNode)
            TSTALLOC(BJT2basePrimeEmitPrimePtr,BJT2basePrimeNode,BJT2emitPrimeNode)
            TSTALLOC(BJT2emitPrimeEmitPtr,BJT2emitPrimeNode,BJT2emitNode)
            TSTALLOC(BJT2emitPrimeColPrimePtr,BJT2emitPrimeNode,BJT2colPrimeNode)
            TSTALLOC(BJT2emitPrimeBasePrimePtr,BJT2emitPrimeNode,BJT2basePrimeNode)
            TSTALLOC(BJT2colColPtr,BJT2colNode,BJT2colNode)
            TSTALLOC(BJT2baseBasePtr,BJT2baseNode,BJT2baseNode)
            TSTALLOC(BJT2emitEmitPtr,BJT2emitNode,BJT2emitNode)
            TSTALLOC(BJT2colPrimeColPrimePtr,BJT2colPrimeNode,BJT2colPrimeNode)
            TSTALLOC(BJT2basePrimeBasePrimePtr,BJT2basePrimeNode,BJT2basePrimeNode)
            TSTALLOC(BJT2emitPrimeEmitPrimePtr,BJT2emitPrimeNode,BJT2emitPrimeNode)
            TSTALLOC(BJT2substSubstPtr,BJT2substNode,BJT2substNode)
            if (model -> BJT2subs == LATERAL) {
              here -> BJT2substConNode = here -> BJT2basePrimeNode;
              here -> BJT2substConSubstConPtr =
                                              here -> BJT2basePrimeBasePrimePtr;
            } else {
              here -> BJT2substConNode = here -> BJT2colPrimeNode;
              here -> BJT2substConSubstConPtr = here -> BJT2colPrimeColPrimePtr;
            };
            TSTALLOC(BJT2substConSubstPtr,BJT2substConNode,BJT2substNode)
            TSTALLOC(BJT2substSubstConPtr,BJT2substNode,BJT2substConNode)
            TSTALLOC(BJT2baseColPrimePtr,BJT2baseNode,BJT2colPrimeNode)
            TSTALLOC(BJT2colPrimeBasePtr,BJT2colPrimeNode,BJT2baseNode)
        }
    }
    return(OK);
}

int
BJT2unsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    BJT2model *model;
    BJT2instance *here;

    for (model = (BJT2model *)inModel; model != NULL;
	    model = model->BJT2nextModel)
    {
        for (here = model->BJT2instances; here != NULL;
                here=here->BJT2nextInstance)
	{
	    if (here->BJT2colPrimeNode
		    && here->BJT2colPrimeNode != here->BJT2colNode)
	    {
		CKTdltNNum(ckt, here->BJT2colPrimeNode);
		here->BJT2colPrimeNode = 0;
	    }
	    if (here->BJT2basePrimeNode
		    && here->BJT2basePrimeNode != here->BJT2baseNode)
	    {
		CKTdltNNum(ckt, here->BJT2basePrimeNode);
		here->BJT2basePrimeNode = 0;
	    }
	    if (here->BJT2emitPrimeNode
		    && here->BJT2emitPrimeNode != here->BJT2emitNode)
	    {
		CKTdltNNum(ckt, here->BJT2emitPrimeNode);
		here->BJT2emitPrimeNode = 0;
	    }
	}
    }
    return OK;
}

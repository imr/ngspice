/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
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
#include "bjtdefs.h"
#include "const.h"
#include "sperror.h"
#include "ifsim.h"
#include "suffix.h"

int
BJTsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
  /* load the BJT structure with those pointers needed later 
  * for fast matrix loading 
  */
{
    BJTmodel *model = (BJTmodel*)inModel;
    BJTinstance *here;
    int error;
    CKTnode *tmp;

    /*  loop through all the diode models */
    for( ; model != NULL; model = model->BJTnextModel ) {

        if(model->BJTtype != NPN && model->BJTtype != PNP) {
            model->BJTtype = NPN;
        }
        if(!model->BJTsatCurGiven) {
            model->BJTsatCur = 1e-16;
        }
        if(!model->BJTbetaFGiven) {
            model->BJTbetaF = 100;
        }
        if(!model->BJTemissionCoeffFGiven) {
            model->BJTemissionCoeffF = 1;
        }
        if(!model->BJTleakBEemissionCoeffGiven) {
            model->BJTleakBEemissionCoeff = 1.5;
        }
        if(!model->BJTbetaRGiven) {
            model->BJTbetaR = 1;
        }
        if(!model->BJTemissionCoeffRGiven) {
            model->BJTemissionCoeffR = 1;
        }
        if(!model->BJTleakBCemissionCoeffGiven) {
            model->BJTleakBCemissionCoeff = 2;
        }
        if(!model->BJTbaseResistGiven) {
            model->BJTbaseResist = 0;
        }
        if(!model->BJTemitterResistGiven) {
            model->BJTemitterResist = 0;
        }
        if(!model->BJTcollectorResistGiven) {
            model->BJTcollectorResist = 0;
        }
        if(!model->BJTdepletionCapBEGiven) {
            model->BJTdepletionCapBE = 0;
        }
        if(!model->BJTpotentialBEGiven) {
            model->BJTpotentialBE = .75;
        }
        if(!model->BJTjunctionExpBEGiven) {
            model->BJTjunctionExpBE = .33;
        }
        if(!model->BJTtransitTimeFGiven) {
            model->BJTtransitTimeF = 0;
        }
        if(!model->BJTtransitTimeBiasCoeffFGiven) {
            model->BJTtransitTimeBiasCoeffF = 0;
        }
        if(!model->BJTtransitTimeHighCurrentFGiven) {
            model->BJTtransitTimeHighCurrentF = 0;
        }
        if(!model->BJTexcessPhaseGiven) {
            model->BJTexcessPhase = 0;
        }
        if(!model->BJTdepletionCapBCGiven) {
            model->BJTdepletionCapBC = 0;
        }
        if(!model->BJTpotentialBCGiven) {
            model->BJTpotentialBC = .75;
        }
        if(!model->BJTjunctionExpBCGiven) {
            model->BJTjunctionExpBC = .33;
        }
        if(!model->BJTbaseFractionBCcapGiven) {
            model->BJTbaseFractionBCcap = 1;
        }
        if(!model->BJTtransitTimeRGiven) {
            model->BJTtransitTimeR = 0;
        }
        if(!model->BJTcapCSGiven) {
            model->BJTcapCS = 0;
        }
        if(!model->BJTpotentialSubstrateGiven) {
            model->BJTpotentialSubstrate = .75;
        }
        if(!model->BJTexponentialSubstrateGiven) {
            model->BJTexponentialSubstrate = 0;
        }
        if(!model->BJTbetaExpGiven) {
            model->BJTbetaExp = 0;
        }
        if(!model->BJTenergyGapGiven) {
            model->BJTenergyGap = 1.11;
        }
        if(!model->BJTtempExpISGiven) {
            model->BJTtempExpIS = 3;
        }
	if(!model->BJTfNcoefGiven) {
	    model->BJTfNcoef = 0;
	}
	if(!model->BJTfNexpGiven) {
	    model->BJTfNexp = 1;
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
        for (here = model->BJTinstances; here != NULL ;
                here=here->BJTnextInstance) {
	    CKTnode *tmpNode;
	    IFuid tmpName;
            
	    if (here->BJTowner != ARCHme)
		goto matrixpointers;
	    
            if(!here->BJTareaGiven) {
                here->BJTarea = 1.0;
            }
	    if(!here->BJTareabGiven) {
                here->BJTareab = here->BJTarea;
            }
	    if(!here->BJTareacGiven) {
                here->BJTareac = here->BJTarea;
            }
	    if(!here->BJTmGiven) {
                here->BJTm = 1.0;
            }
	    
            here->BJTstate = *states;
            *states += BJTnumStates;
            if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN) ){
                *states += 8 * (ckt->CKTsenInfo->SENparms);
            }

matrixpointers:
            if(model->BJTcollectorResist == 0) {
                here->BJTcolPrimeNode = here->BJTcolNode;
            } else if(here->BJTcolPrimeNode == 0) {
                error = CKTmkVolt(ckt,&tmp,here->BJTname,"collector");
                if(error) return(error);
                here->BJTcolPrimeNode = tmp->number;
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
            if(model->BJTbaseResist == 0) {
                here->BJTbasePrimeNode = here->BJTbaseNode;
            } else if(here->BJTbasePrimeNode == 0){
                error = CKTmkVolt(ckt,&tmp,here->BJTname, "base");
                if(error) return(error);
                here->BJTbasePrimeNode = tmp->number;
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
            if(model->BJTemitterResist == 0) {
                here->BJTemitPrimeNode = here->BJTemitNode;
            } else if(here->BJTemitPrimeNode == 0) {
                error = CKTmkVolt(ckt,&tmp,here->BJTname, "emitter");
                if(error) return(error);
                here->BJTemitPrimeNode = tmp->number;
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
            TSTALLOC(BJTcolColPrimePtr,BJTcolNode,BJTcolPrimeNode)
            TSTALLOC(BJTbaseBasePrimePtr,BJTbaseNode,BJTbasePrimeNode)
            TSTALLOC(BJTemitEmitPrimePtr,BJTemitNode,BJTemitPrimeNode)
            TSTALLOC(BJTcolPrimeColPtr,BJTcolPrimeNode,BJTcolNode)
            TSTALLOC(BJTcolPrimeBasePrimePtr,BJTcolPrimeNode,BJTbasePrimeNode)
            TSTALLOC(BJTcolPrimeEmitPrimePtr,BJTcolPrimeNode,BJTemitPrimeNode)
            TSTALLOC(BJTbasePrimeBasePtr,BJTbasePrimeNode,BJTbaseNode)
            TSTALLOC(BJTbasePrimeColPrimePtr,BJTbasePrimeNode,BJTcolPrimeNode)
            TSTALLOC(BJTbasePrimeEmitPrimePtr,BJTbasePrimeNode,BJTemitPrimeNode)
            TSTALLOC(BJTemitPrimeEmitPtr,BJTemitPrimeNode,BJTemitNode)
            TSTALLOC(BJTemitPrimeColPrimePtr,BJTemitPrimeNode,BJTcolPrimeNode)
            TSTALLOC(BJTemitPrimeBasePrimePtr,BJTemitPrimeNode,BJTbasePrimeNode)
            TSTALLOC(BJTcolColPtr,BJTcolNode,BJTcolNode)
            TSTALLOC(BJTbaseBasePtr,BJTbaseNode,BJTbaseNode)
            TSTALLOC(BJTemitEmitPtr,BJTemitNode,BJTemitNode)
            TSTALLOC(BJTcolPrimeColPrimePtr,BJTcolPrimeNode,BJTcolPrimeNode)
            TSTALLOC(BJTbasePrimeBasePrimePtr,BJTbasePrimeNode,BJTbasePrimeNode)
            TSTALLOC(BJTemitPrimeEmitPrimePtr,BJTemitPrimeNode,BJTemitPrimeNode)
            TSTALLOC(BJTsubstSubstPtr,BJTsubstNode,BJTsubstNode)
            TSTALLOC(BJTcolPrimeSubstPtr,BJTcolPrimeNode,BJTsubstNode)
            TSTALLOC(BJTsubstColPrimePtr,BJTsubstNode,BJTcolPrimeNode)
            TSTALLOC(BJTbaseColPrimePtr,BJTbaseNode,BJTcolPrimeNode)
            TSTALLOC(BJTcolPrimeBasePtr,BJTcolPrimeNode,BJTbaseNode)
        }
    }
    return(OK);
}

int
BJTunsetup(inModel,ckt)
    GENmodel *inModel;
    CKTcircuit *ckt;
{
    BJTmodel *model;
    BJTinstance *here;

    for (model = (BJTmodel *)inModel; model != NULL;
	    model = model->BJTnextModel)
    {
        for (here = model->BJTinstances; here != NULL;
                here=here->BJTnextInstance)
	{
	    if (here->BJTcolPrimeNode
		    && here->BJTcolPrimeNode != here->BJTcolNode)
	    {
		CKTdltNNum(ckt, here->BJTcolPrimeNode);
		here->BJTcolPrimeNode = 0;
	    }
	    if (here->BJTbasePrimeNode
		    && here->BJTbasePrimeNode != here->BJTbaseNode)
	    {
		CKTdltNNum(ckt, here->BJTbasePrimeNode);
		here->BJTbasePrimeNode = 0;
	    }
	    if (here->BJTemitPrimeNode
		    && here->BJTemitPrimeNode != here->BJTemitNode)
	    {
		CKTdltNNum(ckt, here->BJTemitPrimeNode);
		here->BJTemitPrimeNode = 0;
	    }
	}
    }
    return OK;
}

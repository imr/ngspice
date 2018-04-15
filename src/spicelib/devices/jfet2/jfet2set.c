/**********
Based on jfetset.c
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

Modified to add PS model and new parameter definitions ( Anthony E. Parker )
   Copyright 1994  Macquarie University, Sydney Australia.
   10 Feb 1994: Added call to jfetparm.h, used JFET_STATE_COUNT
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "jfet2defs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
JFET2setup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
        /* load the diode structure with those pointers needed later 
         * for fast matrix loading 
         */
{
    JFET2model *model = (JFET2model*)inModel;
    JFET2instance *here;
    int error;
    CKTnode *tmp;

    /*  loop through all the diode models */
    for( ; model != NULL; model = JFET2nextModel(model)) {

        if( (model->JFET2type != NJF) && (model->JFET2type != PJF) ) {
            model->JFET2type = NJF;
        }
#define  PARAM(code,id,flag,ref,default,descrip) \
              if(!model->flag) {model->ref = default;}
#include "jfet2parm.h"

        /* loop through all the instances of the model */
        for (here = JFET2instances(model); here != NULL ;
                here=JFET2nextInstance(here)) {
            
            if(!here->JFET2areaGiven) {
                here->JFET2area = 1;
            }
  
            if(!here->JFET2mGiven) {
                here->JFET2m = 1;
            }

            here->JFET2state = *states;
            *states += JFET2numStates;

            if(model->JFET2rs != 0) {
                if(here->JFET2sourcePrimeNode == 0) {
                error = CKTmkVolt(ckt,&tmp,here->JFET2name,"source");
                if(error) return(error);
                here->JFET2sourcePrimeNode = tmp->number;

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
                here->JFET2sourcePrimeNode = here->JFET2sourceNode;
            }
            if(model->JFET2rd != 0) {
                if(here->JFET2drainPrimeNode == 0) {
                error = CKTmkVolt(ckt,&tmp,here->JFET2name,"drain");
                if(error) return(error);
                here->JFET2drainPrimeNode = tmp->number;

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
                here->JFET2drainPrimeNode = here->JFET2drainNode;
            }

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

            TSTALLOC(JFET2drainDrainPrimePtr,JFET2drainNode,JFET2drainPrimeNode);
            TSTALLOC(JFET2gateDrainPrimePtr,JFET2gateNode,JFET2drainPrimeNode);
            TSTALLOC(JFET2gateSourcePrimePtr,JFET2gateNode,JFET2sourcePrimeNode);
            TSTALLOC(JFET2sourceSourcePrimePtr,JFET2sourceNode,JFET2sourcePrimeNode);
            TSTALLOC(JFET2drainPrimeDrainPtr,JFET2drainPrimeNode,JFET2drainNode);
            TSTALLOC(JFET2drainPrimeGatePtr,JFET2drainPrimeNode,JFET2gateNode);
            TSTALLOC(JFET2drainPrimeSourcePrimePtr,JFET2drainPrimeNode,JFET2sourcePrimeNode);
            TSTALLOC(JFET2sourcePrimeGatePtr,JFET2sourcePrimeNode,JFET2gateNode);
            TSTALLOC(JFET2sourcePrimeSourcePtr,JFET2sourcePrimeNode,JFET2sourceNode);
            TSTALLOC(JFET2sourcePrimeDrainPrimePtr,JFET2sourcePrimeNode,JFET2drainPrimeNode);
            TSTALLOC(JFET2drainDrainPtr,JFET2drainNode,JFET2drainNode);
            TSTALLOC(JFET2gateGatePtr,JFET2gateNode,JFET2gateNode);
            TSTALLOC(JFET2sourceSourcePtr,JFET2sourceNode,JFET2sourceNode);
            TSTALLOC(JFET2drainPrimeDrainPrimePtr,JFET2drainPrimeNode,JFET2drainPrimeNode);
            TSTALLOC(JFET2sourcePrimeSourcePrimePtr,JFET2sourcePrimeNode,JFET2sourcePrimeNode);
        }
    }
    return(OK);
}

int
JFET2unsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    JFET2model *model;
    JFET2instance *here;

    for (model = (JFET2model *)inModel; model != NULL;
	    model = JFET2nextModel(model))
    {
        for (here = JFET2instances(model); here != NULL;
                here=JFET2nextInstance(here))
	{
	    if (here->JFET2drainPrimeNode > 0
		    && here->JFET2drainPrimeNode != here->JFET2drainNode)
		CKTdltNNum(ckt, here->JFET2drainPrimeNode);
            here->JFET2drainPrimeNode = 0;

	    if (here->JFET2sourcePrimeNode > 0
		    && here->JFET2sourcePrimeNode != here->JFET2sourceNode)
		CKTdltNNum(ckt, here->JFET2sourcePrimeNode);
            here->JFET2sourcePrimeNode = 0;
	}
    }
    return OK;
}

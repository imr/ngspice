/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
Sydney University mods Copyright(c) 1989 Anthony E. Parker, David J. Skellern
	Laboratory for Communication Science Engineering
	Sydney University Department of Electrical Engineering, Australia
**********/

#include "ngspice.h"
#include "smpdefs.h"
#include "cktdefs.h"
#include "jfetdefs.h"
#include "const.h"
#include "sperror.h"
#include "suffix.h"

int
JFETsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
        /* load the diode structure with those pointers needed later 
         * for fast matrix loading 
         */
{
    JFETmodel *model = (JFETmodel*)inModel;
    JFETinstance *here;
    int error;
    CKTnode *tmp;

    /*  loop through all the diode models */
    for( ; model != NULL; model = model->JFETnextModel ) {

        if( (model->JFETtype != NJF) && (model->JFETtype != PJF) ) {
            model->JFETtype = NJF;
        }
        if(!model->JFETthresholdGiven) {
            model->JFETthreshold = -2;
        }
        if(!model->JFETbetaGiven) {
            model->JFETbeta = 1e-4;
        }
        if(!model->JFETlModulationGiven) {
            model->JFETlModulation = 0;
        }
        if(!model->JFETdrainResistGiven) {
            model->JFETdrainResist = 0;
        }
        if(!model->JFETsourceResistGiven) {
            model->JFETsourceResist = 0;
        }
        if(!model->JFETcapGSGiven) {
            model->JFETcapGS = 0;
        }
        if(!model->JFETcapGDGiven) {
            model->JFETcapGD = 0;
        }
        if(!model->JFETgatePotentialGiven) {
            model->JFETgatePotential = 1;
        }
        if(!model->JFETgateSatCurrentGiven) {
            model->JFETgateSatCurrent = 1e-14;
        }
        if(!model->JFETdepletionCapCoeffGiven) {
            model->JFETdepletionCapCoeff = .5;
        }
	if(!model->JFETfNcoefGiven) {
	    model->JFETfNcoef = 0;
	}
	if(!model->JFETfNexpGiven) {
	    model->JFETfNexp = 1;
	}

	/* Modification for Sydney University JFET model */
	if(!model->JFETbGiven) {
	    model->JFETb = 1.0;
	}
	/* end Sydney University mod */

        if(model->JFETdrainResist != 0) {
            model->JFETdrainConduct = 1/model->JFETdrainResist;
        } else {
            model->JFETdrainConduct = 0;
        }
        if(model->JFETsourceResist != 0) {
            model->JFETsourceConduct = 1/model->JFETsourceResist;
        } else {
            model->JFETsourceConduct = 0;
        }

        /* loop through all the instances of the model */
        for (here = model->JFETinstances; here != NULL ;
                here=here->JFETnextInstance) {
	    if (here->JFETowner != ARCHme) goto matrixpointers;
            
            if(!here->JFETareaGiven) {
                here->JFETarea = 1;
            }
            if(!here->JFETmGiven) {
                here->JFETm = 1;
            }
            here->JFETstate = *states;
            *states += 13;

matrixpointers:
            if(model->JFETsourceResist != 0 && here->JFETsourcePrimeNode==0) {
                error = CKTmkVolt(ckt,&tmp,here->JFETname,"source");
                if(error) return(error);
                here->JFETsourcePrimeNode = tmp->number;
                
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
                here->JFETsourcePrimeNode = here->JFETsourceNode;
            }
            if(model->JFETdrainResist != 0 && here->JFETdrainPrimeNode==0) {
                error = CKTmkVolt(ckt,&tmp,here->JFETname,"drain");
                if(error) return(error);
                here->JFETdrainPrimeNode = tmp->number;
                
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
                here->JFETdrainPrimeNode = here->JFETdrainNode;
            }

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
if((here->ptr = SMPmakeElt(matrix,here->first,here->second))==(double *)NULL){\
    return(E_NOMEM);\
}

            TSTALLOC(JFETdrainDrainPrimePtr,JFETdrainNode,JFETdrainPrimeNode)
            TSTALLOC(JFETgateDrainPrimePtr,JFETgateNode,JFETdrainPrimeNode)
            TSTALLOC(JFETgateSourcePrimePtr,JFETgateNode,JFETsourcePrimeNode)
            TSTALLOC(JFETsourceSourcePrimePtr,JFETsourceNode,
                    JFETsourcePrimeNode)
            TSTALLOC(JFETdrainPrimeDrainPtr,JFETdrainPrimeNode,JFETdrainNode)
            TSTALLOC(JFETdrainPrimeGatePtr,JFETdrainPrimeNode,JFETgateNode)
            TSTALLOC(JFETdrainPrimeSourcePrimePtr,JFETdrainPrimeNode,
                    JFETsourcePrimeNode)
            TSTALLOC(JFETsourcePrimeGatePtr,JFETsourcePrimeNode,JFETgateNode)
            TSTALLOC(JFETsourcePrimeSourcePtr,JFETsourcePrimeNode,
                    JFETsourceNode)
            TSTALLOC(JFETsourcePrimeDrainPrimePtr,JFETsourcePrimeNode,
                    JFETdrainPrimeNode)
            TSTALLOC(JFETdrainDrainPtr,JFETdrainNode,JFETdrainNode)
            TSTALLOC(JFETgateGatePtr,JFETgateNode,JFETgateNode)
            TSTALLOC(JFETsourceSourcePtr,JFETsourceNode,JFETsourceNode)
            TSTALLOC(JFETdrainPrimeDrainPrimePtr,JFETdrainPrimeNode,
                    JFETdrainPrimeNode)
            TSTALLOC(JFETsourcePrimeSourcePrimePtr,JFETsourcePrimeNode,
                    JFETsourcePrimeNode)
        }
    }
    return(OK);
}

int
JFETunsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    JFETmodel *model;
    JFETinstance *here;

    for (model = (JFETmodel *)inModel; model != NULL;
	    model = model->JFETnextModel)
    {
        for (here = model->JFETinstances; here != NULL;
                here=here->JFETnextInstance)
	{
	    if (here->JFETsourcePrimeNode
		    && here->JFETsourcePrimeNode != here->JFETsourceNode)
	    {
		CKTdltNNum(ckt, here->JFETsourcePrimeNode);
		here->JFETsourcePrimeNode = 0;
	    }
	    if (here->JFETdrainPrimeNode
		    && here->JFETdrainPrimeNode != here->JFETdrainNode)
	    {
		CKTdltNNum(ckt, here->JFETdrainPrimeNode);
		here->JFETdrainPrimeNode = 0;
	    }
	}
    }
    return OK;
}

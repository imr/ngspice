/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Paolo Nenzi 2002
File: b3v1getic.c
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "bsim3v1sdef.h"
#include "sperror.h"
#include "suffix.h"


int
BSIM3v1Sgetic(GENmodel *inModel, CKTcircuit *ckt)
{
BSIM3v1Smodel *model = (BSIM3v1Smodel*)inModel;
BSIM3v1Sinstance *here;

    for (; model ; model = model->BSIM3v1SnextModel) 
    {    for (here = model->BSIM3v1Sinstances; here; here = here->BSIM3v1SnextInstance)
	 {    
             
	     if (here->BSIM3v1Sowner != ARCHme) 
	             continue;
		     
             if(!here->BSIM3v1SicVBSGiven) 
	      {  here->BSIM3v1SicVBS = *(ckt->CKTrhs + here->BSIM3v1SbNode) 
				  - *(ckt->CKTrhs + here->BSIM3v1SsNode);
              }
              if (!here->BSIM3v1SicVDSGiven) 
	      {   here->BSIM3v1SicVDS = *(ckt->CKTrhs + here->BSIM3v1SdNode) 
				   - *(ckt->CKTrhs + here->BSIM3v1SsNode);
              }
              if (!here->BSIM3v1SicVGSGiven) 
	      {   here->BSIM3v1SicVGS = *(ckt->CKTrhs + here->BSIM3v1SgNode) 
				   - *(ckt->CKTrhs + here->BSIM3v1SsNode);
              }
         }
    }
    return(OK);
}



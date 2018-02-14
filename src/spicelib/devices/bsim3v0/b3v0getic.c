/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v0getic.c
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3v0def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM3v0getic(GENmodel *inModel, CKTcircuit *ckt)
{
BSIM3v0model *model = (BSIM3v0model*)inModel;
BSIM3v0instance *here;

    for (; model ; model = BSIM3v0nextModel(model)) 
    {    for (here = BSIM3v0instances(model); here; here = BSIM3v0nextInstance(here))
	 {
	      if(!here->BSIM3v0icVBSGiven) 
	      {  here->BSIM3v0icVBS = *(ckt->CKTrhs + here->BSIM3v0bNode) 
				  - *(ckt->CKTrhs + here->BSIM3v0sNode);
              }
              if (!here->BSIM3v0icVDSGiven) 
	      {   here->BSIM3v0icVDS = *(ckt->CKTrhs + here->BSIM3v0dNode) 
				   - *(ckt->CKTrhs + here->BSIM3v0sNode);
              }
              if (!here->BSIM3v0icVGSGiven) 
	      {   here->BSIM3v0icVGS = *(ckt->CKTrhs + here->BSIM3v0gNode) 
				   - *(ckt->CKTrhs + here->BSIM3v0sNode);
              }
         }
    }
    return(OK);
}



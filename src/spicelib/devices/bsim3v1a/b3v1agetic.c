/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Paolo Nenzi 2002
File: b3v1agetic.c
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "bsim3v1adef.h"
#include "sperror.h"
#include "suffix.h"


int
BSIM3v1Agetic(GENmodel *inModel, CKTcircuit *ckt)
{
BSIM3v1Amodel *model = (BSIM3v1Amodel*)inModel;
BSIM3v1Ainstance *here;

    for (; model ; model = model->BSIM3v1AnextModel) 
    {    for (here = model->BSIM3v1Ainstances; here; here = here->BSIM3v1AnextInstance)
	 {    
	 
	      if (here->BSIM3v1Aowner != ARCHme)
                      continue;

	      if(!here->BSIM3v1AicVBSGiven) 
	      {  here->BSIM3v1AicVBS = *(ckt->CKTrhs + here->BSIM3v1AbNode) 
				  - *(ckt->CKTrhs + here->BSIM3v1AsNode);
              }
              if (!here->BSIM3v1AicVDSGiven) 
	      {   here->BSIM3v1AicVDS = *(ckt->CKTrhs + here->BSIM3v1AdNode) 
				   - *(ckt->CKTrhs + here->BSIM3v1AsNode);
              }
              if (!here->BSIM3v1AicVGSGiven) 
	      {   here->BSIM3v1AicVGS = *(ckt->CKTrhs + here->BSIM3v1AgNode) 
				   - *(ckt->CKTrhs + here->BSIM3v1AsNode);
              }
         }
    }
    return(OK);
}



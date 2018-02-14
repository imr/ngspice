/**********
 * Copyright 1990 Regents of the University of California. All rights reserved.
 * File: b3v1getic.c
 * Author: 1995 Min-Chie Jeng and Mansun Chan. 
 * Modified by Paolo Nenzi 2002
 **********/
 
/* 
 * Release Notes: 
 * BSIM3v3.1,   Released by yuhua  96/12/08
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3v1def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM3v1getic(GENmodel *inModel, CKTcircuit *ckt)
{
BSIM3v1model *model = (BSIM3v1model*)inModel;
BSIM3v1instance *here;

    for (; model ; model = BSIM3v1nextModel(model)) 
    {    for (here = BSIM3v1instances(model); here; here = BSIM3v1nextInstance(here))
	 {
	      if(!here->BSIM3v1icVBSGiven) 
	      {  here->BSIM3v1icVBS = *(ckt->CKTrhs + here->BSIM3v1bNode) 
				  - *(ckt->CKTrhs + here->BSIM3v1sNode);
              }
              if (!here->BSIM3v1icVDSGiven) 
	      {   here->BSIM3v1icVDS = *(ckt->CKTrhs + here->BSIM3v1dNode) 
				   - *(ckt->CKTrhs + here->BSIM3v1sNode);
              }
              if (!here->BSIM3v1icVGSGiven) 
	      {   here->BSIM3v1icVGS = *(ckt->CKTrhs + here->BSIM3v1gNode) 
				   - *(ckt->CKTrhs + here->BSIM3v1sNode);
              }
         }
    }
    return(OK);
}

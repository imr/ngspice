/**** BSIM3v3.3.0, Released by Xuemei Xi 07/29/2005 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b3getic.c of BSIM3v3.3.0
 * Author: 1995 Min-Chie Jeng and Mansun Chan.
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001  Xuemei Xi
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM3getic(
GENmodel *inModel,
CKTcircuit *ckt)
{
BSIM3model *model = (BSIM3model*)inModel;
BSIM3instance *here;

    for (; model ; model = BSIM3nextModel(model)) 
    {    for (here = BSIM3instances(model); here; here = BSIM3nextInstance(here))
	 {
              if (!here->BSIM3icVBSGiven) 
	      {  here->BSIM3icVBS = *(ckt->CKTrhs + here->BSIM3bNode) 
				  - *(ckt->CKTrhs + here->BSIM3sNode);
              }
              if (!here->BSIM3icVDSGiven) 
	      {   here->BSIM3icVDS = *(ckt->CKTrhs + here->BSIM3dNode) 
				   - *(ckt->CKTrhs + here->BSIM3sNode);
              }
              if (!here->BSIM3icVGSGiven) 
	      {   here->BSIM3icVGS = *(ckt->CKTrhs + here->BSIM3gNode) 
				   - *(ckt->CKTrhs + here->BSIM3sNode);
              }
         }
    }
    return(OK);
}

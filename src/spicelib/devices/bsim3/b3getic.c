/**** BSIM3v3.2.4, Released by Xuemei Xi 12/14/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b3getic.c of BSIM3v3.2.4
 * Author: 1995 Min-Chie Jeng and Mansun Chan.
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001  Xuemei Xi
 **********/

#include "ngspice.h"
#include "cktdefs.h"
#include "bsim3def.h"
#include "sperror.h"
#include "suffix.h"


int
BSIM3getic (GENmodel *inModel, CKTcircuit *ckt)
{
BSIM3model *model = (BSIM3model*)inModel;
BSIM3instance *here;

    for (; model ; model = model->BSIM3nextModel) 
    {    for (here = model->BSIM3instances; here; here = here->BSIM3nextInstance)
	 {   	
 	      if (here->BSIM3owner != ARCHme)
			continue;

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



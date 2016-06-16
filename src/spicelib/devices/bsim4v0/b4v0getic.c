/**** BSIM4v0.0.0, Released by Weidong Liu 3/24/2000 ****/

/**********
 * Copyright 2000 Regents of the University of California. All rights reserved.
 * File: b4getic.c of BSIM4v0.0.0.
 * Authors: Weidong Liu, Kanyu M. Cao, Xiaodong Jin, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v0def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM4v0getic(inModel,ckt)
GENmodel *inModel;
CKTcircuit *ckt;
{
BSIM4v0model *model = (BSIM4v0model*)inModel;
BSIM4v0instance *here;

    for (; model ; model = model->BSIM4v0nextModel) 
    {    for (here = model->BSIM4v0instances; here; here = here->BSIM4v0nextInstance)
	 {    if (!here->BSIM4v0icVDSGiven) 
	      {   here->BSIM4v0icVDS = *(ckt->CKTrhs + here->BSIM4v0dNode) 
				   - *(ckt->CKTrhs + here->BSIM4v0sNode);
              }
              if (!here->BSIM4v0icVGSGiven) 
	      {   here->BSIM4v0icVGS = *(ckt->CKTrhs + here->BSIM4v0gNodeExt) 
				   - *(ckt->CKTrhs + here->BSIM4v0sNode);
              }
              if(!here->BSIM4v0icVBSGiven)
              {  here->BSIM4v0icVBS = *(ckt->CKTrhs + here->BSIM4v0bNode)
                                  - *(ckt->CKTrhs + here->BSIM4v0sNode);
              }
         }
    }
    return(OK);
}

/**** BSIM4.5.0 Released by Xuemei (Jane) Xi 07/29/2005 ****/

/**********
 * Copyright 2005 Regents of the University of California. All rights reserved.
 * File: b4getic.c of BSIM4.5.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v5def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM4v5getic(
GENmodel *inModel,
CKTcircuit *ckt)
{
BSIM4v5model *model = (BSIM4v5model*)inModel;
BSIM4v5instance *here;

    for (; model ; model = BSIM4v5nextModel(model)) 
    {    for (here = BSIM4v5instances(model); here; here = BSIM4v5nextInstance(here))
	      {
	           if (!here->BSIM4v5icVDSGiven) 
	      {   here->BSIM4v5icVDS = *(ckt->CKTrhs + here->BSIM4v5dNode) 
				   - *(ckt->CKTrhs + here->BSIM4v5sNode);
              }
              if (!here->BSIM4v5icVGSGiven) 
	      {   here->BSIM4v5icVGS = *(ckt->CKTrhs + here->BSIM4v5gNodeExt) 
				   - *(ckt->CKTrhs + here->BSIM4v5sNode);
              }
              if(!here->BSIM4v5icVBSGiven)
              {  here->BSIM4v5icVBS = *(ckt->CKTrhs + here->BSIM4v5bNode)
                                  - *(ckt->CKTrhs + here->BSIM4v5sNode);
              }
         }
    }
    return(OK);
}

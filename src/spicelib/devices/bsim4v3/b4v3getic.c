/**** BSIM4.3.0 Released by Xuemei (Jane) Xi 05/09/2003 ****/

/**********
 * Copyright 2003 Regents of the University of California. All rights reserved.
 * File: b4v3getic.c of BSIM4.3.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "bsim4v3def.h"
#include "sperror.h"


int
BSIM4v3getic(
GENmodel *inModel,
CKTcircuit *ckt)
{
BSIM4v3model *model = (BSIM4v3model*)inModel;
BSIM4v3instance *here;

    for (; model ; model = model->BSIM4v3nextModel) 
    {    for (here = model->BSIM4v3instances; here; here = here->BSIM4v3nextInstance)
         {    if (here->BSIM4v3owner != ARCHme) continue;
	      if (!here->BSIM4v3icVDSGiven) 
	      {   here->BSIM4v3icVDS = *(ckt->CKTrhs + here->BSIM4v3dNode) 
				   - *(ckt->CKTrhs + here->BSIM4v3sNode);
              }
              if (!here->BSIM4v3icVGSGiven) 
	      {   here->BSIM4v3icVGS = *(ckt->CKTrhs + here->BSIM4v3gNodeExt) 
				   - *(ckt->CKTrhs + here->BSIM4v3sNode);
              }
              if(!here->BSIM4v3icVBSGiven)
              {  here->BSIM4v3icVBS = *(ckt->CKTrhs + here->BSIM4v3bNode)
                                  - *(ckt->CKTrhs + here->BSIM4v3sNode);
              }
         }
    }
    return(OK);
}

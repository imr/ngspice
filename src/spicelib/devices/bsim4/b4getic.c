/**** BSIM4.4.0  Released by Xuemei (Jane) Xi 03/04/2004 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b4getic.c of BSIM4.4.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice.h"
#include "cktdefs.h"
#include "bsim4def.h"
#include "sperror.h"
#include "suffix.h"


int
BSIM4getic(inModel,ckt)
GENmodel *inModel;
CKTcircuit *ckt;
{
BSIM4model *model = (BSIM4model*)inModel;
BSIM4instance *here;

    for (; model ; model = model->BSIM4nextModel) 
    {    for (here = model->BSIM4instances; here; here = here->BSIM4nextInstance)
	       {   if (here->BSIM4owner != ARCHme) continue;
	           if (!here->BSIM4icVDSGiven) 
	      {   here->BSIM4icVDS = *(ckt->CKTrhs + here->BSIM4dNode) 
				   - *(ckt->CKTrhs + here->BSIM4sNode);
              }
              if (!here->BSIM4icVGSGiven) 
	      {   here->BSIM4icVGS = *(ckt->CKTrhs + here->BSIM4gNodeExt) 
				   - *(ckt->CKTrhs + here->BSIM4sNode);
              }
              if(!here->BSIM4icVBSGiven)
              {  here->BSIM4icVBS = *(ckt->CKTrhs + here->BSIM4bNode)
                                  - *(ckt->CKTrhs + here->BSIM4sNode);
              }
         }
    }
    return(OK);
}

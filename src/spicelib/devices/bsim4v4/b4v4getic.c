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
#include "bsim4v4def.h"
#include "sperror.h"
#include "suffix.h"


int
BSIM4V4getic(inModel,ckt)
GENmodel *inModel;
CKTcircuit *ckt;
{
BSIM4V4model *model = (BSIM4V4model*)inModel;
BSIM4V4instance *here;

    for (; model ; model = model->BSIM4V4nextModel) 
    {    for (here = model->BSIM4V4instances; here; here = here->BSIM4V4nextInstance)
	       {   if (here->BSIM4V4owner != ARCHme) continue;
	           if (!here->BSIM4V4icVDSGiven) 
	      {   here->BSIM4V4icVDS = *(ckt->CKTrhs + here->BSIM4V4dNode) 
				   - *(ckt->CKTrhs + here->BSIM4V4sNode);
              }
              if (!here->BSIM4V4icVGSGiven) 
	      {   here->BSIM4V4icVGS = *(ckt->CKTrhs + here->BSIM4V4gNodeExt) 
				   - *(ckt->CKTrhs + here->BSIM4V4sNode);
              }
              if(!here->BSIM4V4icVBSGiven)
              {  here->BSIM4V4icVBS = *(ckt->CKTrhs + here->BSIM4V4bNode)
                                  - *(ckt->CKTrhs + here->BSIM4V4sNode);
              }
         }
    }
    return(OK);
}

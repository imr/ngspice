/**** BSIM4.4.0  Released by Xuemei (Jane) Xi 03/04/2004 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b4getic.c of BSIM4.4.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v4def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM4v4getic(
GENmodel *inModel,
CKTcircuit *ckt)
{
BSIM4v4model *model = (BSIM4v4model*)inModel;
BSIM4v4instance *here;

    for (; model ; model = model->BSIM4v4nextModel) 
    {    for (here = model->BSIM4v4instances; here; here = here->BSIM4v4nextInstance)
	       {
	           if (!here->BSIM4v4icVDSGiven) 
	      {   here->BSIM4v4icVDS = *(ckt->CKTrhs + here->BSIM4v4dNode) 
				   - *(ckt->CKTrhs + here->BSIM4v4sNode);
              }
              if (!here->BSIM4v4icVGSGiven) 
	      {   here->BSIM4v4icVGS = *(ckt->CKTrhs + here->BSIM4v4gNodeExt) 
				   - *(ckt->CKTrhs + here->BSIM4v4sNode);
              }
              if(!here->BSIM4v4icVBSGiven)
              {  here->BSIM4v4icVBS = *(ckt->CKTrhs + here->BSIM4v4bNode)
                                  - *(ckt->CKTrhs + here->BSIM4v4sNode);
              }
         }
    }
    return(OK);
}

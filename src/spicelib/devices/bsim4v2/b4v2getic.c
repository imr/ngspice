/**** BSIM4.2.1, Released by Xuemei Xi 10/05/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b4getic.c of BSIM4.2.1.
 * Author: 2000 Weidong Liu
 * Authors: Xuemei Xi, Kanyu M. Cao, Hui Wan, Mansun Chan, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "bsim4v2def.h"
#include "sperror.h"


int
BSIM4v2getic(inModel,ckt)
GENmodel *inModel;
CKTcircuit *ckt;
{
BSIM4v2model *model = (BSIM4v2model*)inModel;
BSIM4v2instance *here;

    for (; model ; model = model->BSIM4v2nextModel) 
    {    for (here = model->BSIM4v2instances; here; here = here->BSIM4v2nextInstance)
	 {   if (here->BSIM4v2owner != ARCHme) continue;
	     if (!here->BSIM4v2icVDSGiven) 
	      {   here->BSIM4v2icVDS = *(ckt->CKTrhs + here->BSIM4v2dNode) 
				   - *(ckt->CKTrhs + here->BSIM4v2sNode);
              }
              if (!here->BSIM4v2icVGSGiven) 
	      {   here->BSIM4v2icVGS = *(ckt->CKTrhs + here->BSIM4v2gNodeExt) 
				   - *(ckt->CKTrhs + here->BSIM4v2sNode);
              }
              if(!here->BSIM4v2icVBSGiven)
              {  here->BSIM4v2icVBS = *(ckt->CKTrhs + here->BSIM4v2bNode)
                                  - *(ckt->CKTrhs + here->BSIM4v2sNode);
              }
         }
    }
    return(OK);
}

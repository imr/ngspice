/**** BSIM4.6.2 Released by Wenwei Yang 07/31/2008****/

/**********
 * Copyright 2006 Regents of the University of California. All rights reserved.
 * File: b4getic.c of BSIM4.6.2.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Authors: 2006- Mohan Dunga, Ali Niknejad, Chenming Hu
 * Authors: 2007- Mohan Dunga, Wenwei Yang, Ali Niknejad, Chenming Hu
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v6def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM4v6getic(
GENmodel *inModel,
CKTcircuit *ckt)
{
BSIM4v6model *model = (BSIM4v6model*)inModel;
BSIM4v6instance *here;

    for (; model ; model = BSIM4v6nextModel(model)) 
    {    for (here = BSIM4v6instances(model); here; here = BSIM4v6nextInstance(here))
          {
	      if (!here->BSIM4v6icVDSGiven) 
	      {   here->BSIM4v6icVDS = *(ckt->CKTrhs + here->BSIM4v6dNode) 
				   - *(ckt->CKTrhs + here->BSIM4v6sNode);
              }
              if (!here->BSIM4v6icVGSGiven) 
	      {   here->BSIM4v6icVGS = *(ckt->CKTrhs + here->BSIM4v6gNodeExt) 
				   - *(ckt->CKTrhs + here->BSIM4v6sNode);
              }
              if(!here->BSIM4v6icVBSGiven)
              {  here->BSIM4v6icVBS = *(ckt->CKTrhs + here->BSIM4v6bNode)
                                  - *(ckt->CKTrhs + here->BSIM4v6sNode);
              }
         }
    }
    return(OK);
}

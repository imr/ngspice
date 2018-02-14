/**** BSIM3v3.2.4, Released by Xuemei Xi 12/14/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b3getic.c of BSIM3v3.2.4
 * Author: 1995 Min-Chie Jeng and Mansun Chan.
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001  Xuemei Xi
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3v32def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM3v32getic (GENmodel *inModel, CKTcircuit *ckt)
{
BSIM3v32model *model = (BSIM3v32model*)inModel;
BSIM3v32instance *here;

    for (; model ; model = BSIM3v32nextModel(model))
    {    for (here = BSIM3v32instances(model); here; here = BSIM3v32nextInstance(here))
         {
              if (!here->BSIM3v32icVBSGiven)
              {  here->BSIM3v32icVBS = *(ckt->CKTrhs + here->BSIM3v32bNode)
                                  - *(ckt->CKTrhs + here->BSIM3v32sNode);
              }
              if (!here->BSIM3v32icVDSGiven)
              {   here->BSIM3v32icVDS = *(ckt->CKTrhs + here->BSIM3v32dNode)
                                   - *(ckt->CKTrhs + here->BSIM3v32sNode);
              }
              if (!here->BSIM3v32icVGSGiven)
              {   here->BSIM3v32icVGS = *(ckt->CKTrhs + here->BSIM3v32gNode)
                                   - *(ckt->CKTrhs + here->BSIM3v32sNode);
              }
         }
    }
    return(OK);
}



/***  B4SOI 12/16/2010 Released by Tanvir Morshed   ***/


/**********
 * Copyright 2010 Regents of the University of California.  All rights reserved.
 * Authors: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
 * Authors: 1999-2004 Pin Su, Hui Wan, Wei Jin, b3soigetic.c
 * Authors: 2005- Hui Wan, Xuemei Xi, Ali Niknejad, Chenming Hu.
 * Authors: 2009- Wenwei Yang, Chung-Hsun Lin, Ali Niknejad, Chenming Hu.
 * File: b4soigetic.c
 * Modified by Hui Wan, Xuemei Xi 11/30/2005
 * Modified by Wenwei Yang, Chung-Hsun Lin, Darsen Lu 03/06/2009
 * Modified by Tanvir Morshed 09/22/2009
 * Modified by Tanvir Morshed 12/31/2009
 **********/

#include "ngspice/ngspice.h"

#include "ngspice/cktdefs.h"
#include "b4soidef.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
B4SOIgetic(
GENmodel *inModel,
CKTcircuit *ckt)
{
B4SOImodel *model = (B4SOImodel*)inModel;
B4SOIinstance *here;

    for (; model ; model = B4SOInextModel(model)) 
    {    for (here = B4SOIinstances(model); here; here = B4SOInextInstance(here))
         {
              if(!here->B4SOIicVBSGiven) 
              {  here->B4SOIicVBS = *(ckt->CKTrhs + here->B4SOIbNode) 
                                  - *(ckt->CKTrhs + here->B4SOIsNode);
              }
              if (!here->B4SOIicVDSGiven) 
              {   here->B4SOIicVDS = *(ckt->CKTrhs + here->B4SOIdNode) 
                                   - *(ckt->CKTrhs + here->B4SOIsNode);
              }
              if (!here->B4SOIicVGSGiven) 
              {   here->B4SOIicVGS = *(ckt->CKTrhs + here->B4SOIgNode) 
                                   - *(ckt->CKTrhs + here->B4SOIsNode);
              }
              if (!here->B4SOIicVESGiven) 
              {   here->B4SOIicVES = *(ckt->CKTrhs + here->B4SOIeNode) 
                                   - *(ckt->CKTrhs + here->B4SOIsNode);
              }
              if (!here->B4SOIicVPSGiven) 
              {   here->B4SOIicVPS = *(ckt->CKTrhs + here->B4SOIpNode) 
                                   - *(ckt->CKTrhs + here->B4SOIsNode);
              }
         }
    }
    return(OK);
}



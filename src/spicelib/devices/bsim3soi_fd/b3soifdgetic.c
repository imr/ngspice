/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soifdgetic.c          98/5/01
**********/


#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "b3soifddef.h"
#include "sperror.h"
#include "suffix.h"


int
B3SOIFDgetic(inModel,ckt)
GENmodel *inModel;
CKTcircuit *ckt;
{
B3SOIFDmodel *model = (B3SOIFDmodel*)inModel;
B3SOIFDinstance *here;

    for (; model ; model = model->B3SOIFDnextModel) 
    {    for (here = model->B3SOIFDinstances; here; here = here->B3SOIFDnextInstance)
	 {    if(!here->B3SOIFDicVBSGiven) 
	      {  here->B3SOIFDicVBS = *(ckt->CKTrhs + here->B3SOIFDbNode) 
				  - *(ckt->CKTrhs + here->B3SOIFDsNode);
              }
              if (!here->B3SOIFDicVDSGiven) 
	      {   here->B3SOIFDicVDS = *(ckt->CKTrhs + here->B3SOIFDdNode) 
				   - *(ckt->CKTrhs + here->B3SOIFDsNode);
              }
              if (!here->B3SOIFDicVGSGiven) 
	      {   here->B3SOIFDicVGS = *(ckt->CKTrhs + here->B3SOIFDgNode) 
				   - *(ckt->CKTrhs + here->B3SOIFDsNode);
              }
              if (!here->B3SOIFDicVESGiven) 
	      {   here->B3SOIFDicVES = *(ckt->CKTrhs + here->B3SOIFDeNode) 
				   - *(ckt->CKTrhs + here->B3SOIFDsNode);
              }
              if (!here->B3SOIFDicVPSGiven) 
	      {   here->B3SOIFDicVPS = *(ckt->CKTrhs + here->B3SOIFDpNode) 
				   - *(ckt->CKTrhs + here->B3SOIFDsNode);
              }
         }
    }
    return(OK);
}



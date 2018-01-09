/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
Modified by Paolo Nenzi 2002
File: b3soifdgetic.c          98/5/01
**********/

/*
 * Revision 2.1  99/9/27 Pin Su 
 * BSIMFD2.1 release
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b3soifddef.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
B3SOIFDgetic(GENmodel *inModel, CKTcircuit *ckt)
{
B3SOIFDmodel *model = (B3SOIFDmodel*)inModel;
B3SOIFDinstance *here;

    for (; model ; model = B3SOIFDnextModel(model)) 
    {    for (here = B3SOIFDinstances(model); here; here = B3SOIFDnextInstance(here))
	 {
	      if(!here->B3SOIFDicVBSGiven) 
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



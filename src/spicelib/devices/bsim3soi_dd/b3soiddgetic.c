/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soiddgetic.c          98/5/01
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.1  99/9/27 Pin Su 
 * BSIMDD2.1 release
 */
 
#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b3soidddef.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
B3SOIDDgetic(GENmodel *inModel, CKTcircuit *ckt)
{
B3SOIDDmodel *model = (B3SOIDDmodel*)inModel;
B3SOIDDinstance *here;

    for (; model ; model = B3SOIDDnextModel(model)) 
    {    for (here = B3SOIDDinstances(model); here; here = B3SOIDDnextInstance(here))
	 {
	      if(!here->B3SOIDDicVBSGiven) 
	      {  here->B3SOIDDicVBS = *(ckt->CKTrhs + here->B3SOIDDbNode) 
				  - *(ckt->CKTrhs + here->B3SOIDDsNode);
              }
              if (!here->B3SOIDDicVDSGiven) 
	      {   here->B3SOIDDicVDS = *(ckt->CKTrhs + here->B3SOIDDdNode) 
				   - *(ckt->CKTrhs + here->B3SOIDDsNode);
              }
              if (!here->B3SOIDDicVGSGiven) 
	      {   here->B3SOIDDicVGS = *(ckt->CKTrhs + here->B3SOIDDgNode) 
				   - *(ckt->CKTrhs + here->B3SOIDDsNode);
              }
              if (!here->B3SOIDDicVESGiven) 
	      {   here->B3SOIDDicVES = *(ckt->CKTrhs + here->B3SOIDDeNode) 
				   - *(ckt->CKTrhs + here->B3SOIDDsNode);
              }
              if (!here->B3SOIDDicVPSGiven) 
	      {   here->B3SOIDDicVPS = *(ckt->CKTrhs + here->B3SOIDDpNode) 
				   - *(ckt->CKTrhs + here->B3SOIDDsNode);
              }
         }
    }
    return(OK);
}



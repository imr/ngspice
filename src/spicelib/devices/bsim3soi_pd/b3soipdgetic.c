/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soipdgetic.c          98/5/01
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.2.3  02/3/5  Pin Su 
 * BSIMPD2.2.3 release
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "b3soipddef.h"
#include "sperror.h"
#include "suffix.h"


int
B3SOIPDgetic(GENmodel *inModel, CKTcircuit *ckt)
{
B3SOIPDmodel *model = (B3SOIPDmodel*)inModel;
B3SOIPDinstance *here;

    for (; model ; model = model->B3SOIPDnextModel) 
    {    for (here = model->B3SOIPDinstances; here; here = here->B3SOIPDnextInstance)
	 {    
              
	      if (here->B3SOIPDowner != ARCHme)
                      continue;
	 
	      if(!here->B3SOIPDicVBSGiven) 
	      {  here->B3SOIPDicVBS = *(ckt->CKTrhs + here->B3SOIPDbNode) 
				  - *(ckt->CKTrhs + here->B3SOIPDsNode);
              }
              if (!here->B3SOIPDicVDSGiven) 
	      {   here->B3SOIPDicVDS = *(ckt->CKTrhs + here->B3SOIPDdNode) 
				   - *(ckt->CKTrhs + here->B3SOIPDsNode);
              }
              if (!here->B3SOIPDicVGSGiven) 
	      {   here->B3SOIPDicVGS = *(ckt->CKTrhs + here->B3SOIPDgNode) 
				   - *(ckt->CKTrhs + here->B3SOIPDsNode);
              }
              if (!here->B3SOIPDicVESGiven) 
	      {   here->B3SOIPDicVES = *(ckt->CKTrhs + here->B3SOIPDeNode) 
				   - *(ckt->CKTrhs + here->B3SOIPDsNode);
              }
              if (!here->B3SOIPDicVPSGiven) 
	      {   here->B3SOIPDicVPS = *(ckt->CKTrhs + here->B3SOIPDpNode) 
				   - *(ckt->CKTrhs + here->B3SOIPDsNode);
              }
         }
    }
    return(OK);
}



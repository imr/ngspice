/* $Id$  */
/* 
$Log$
Revision 1.1.1.1  2000-04-27 20:03:59  pnenzi
Imported sources

 * Revision 3.1  96/12/08  19:54:55  yuhua
 * BSIM3v3.1
 * release
 * 
*/
static char rcsid[] = "$Id$";

/*************************************/

/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v1getic.c
**********/

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "bsim3v1def.h"
#include "sperror.h"
#include "suffix.h"


int
BSIM3V1getic(inModel,ckt)
GENmodel *inModel;
CKTcircuit *ckt;
{
BSIM3V1model *model = (BSIM3V1model*)inModel;
BSIM3V1instance *here;

    for (; model ; model = model->BSIM3V1nextModel) 
    {    for (here = model->BSIM3V1instances; here; here = here->BSIM3V1nextInstance)
	 {    
             if (here->BSIM3V1owner != ARCHme) continue;
             if(!here->BSIM3V1icVBSGiven) 
	      {  here->BSIM3V1icVBS = *(ckt->CKTrhs + here->BSIM3V1bNode) 
				  - *(ckt->CKTrhs + here->BSIM3V1sNode);
              }
              if (!here->BSIM3V1icVDSGiven) 
	      {   here->BSIM3V1icVDS = *(ckt->CKTrhs + here->BSIM3V1dNode) 
				   - *(ckt->CKTrhs + here->BSIM3V1sNode);
              }
              if (!here->BSIM3V1icVGSGiven) 
	      {   here->BSIM3V1icVGS = *(ckt->CKTrhs + here->BSIM3V1gNode) 
				   - *(ckt->CKTrhs + here->BSIM3V1sNode);
              }
         }
    }
    return(OK);
}



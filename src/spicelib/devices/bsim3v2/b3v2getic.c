/* $Id$  */
/*
 $Log$
 Revision 1.1  2000-04-27 20:03:59  pnenzi
 Initial revision

 * Revision 3.2 1998/6/16  18:00:00  Weidong 
 * BSIM3v3.2 release
 *
*/
static char rcsid[] = "$Id$";

/*************************************/

/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v2getic.c
**********/

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "bsim3v2def.h"
#include "sperror.h"
#include "suffix.h"


int
BSIM3V2getic(inModel,ckt)
GENmodel *inModel;
CKTcircuit *ckt;
{
BSIM3V2model *model = (BSIM3V2model*)inModel;
BSIM3V2instance *here;

    for (; model ; model = model->BSIM3V2nextModel) 
    {    for (here = model->BSIM3V2instances; here; here = here->BSIM3V2nextInstance)
	 {     
              if (here->BSIM3V2owner != ARCHme) continue; 
              if(!here->BSIM3V2icVBSGiven) 
	      {  here->BSIM3V2icVBS = *(ckt->CKTrhs + here->BSIM3V2bNode) 
				  - *(ckt->CKTrhs + here->BSIM3V2sNode);
              }
              if (!here->BSIM3V2icVDSGiven) 
	      {   here->BSIM3V2icVDS = *(ckt->CKTrhs + here->BSIM3V2dNode) 
				   - *(ckt->CKTrhs + here->BSIM3V2sNode);
              }
              if (!here->BSIM3V2icVGSGiven) 
	      {   here->BSIM3V2icVGS = *(ckt->CKTrhs + here->BSIM3V2gNode) 
				   - *(ckt->CKTrhs + here->BSIM3V2sNode);
              }
         }
    }
    return(OK);
}



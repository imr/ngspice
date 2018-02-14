/**********
 * Copyright 1990 Regents of the University of California. All rights reserved.
 * File: b3v1cvtest.c
 * Author: 1995 Min-Chie Jeng and Mansun Chan. 
 * Modified by Paolo Nenzi 2002
 **********/
 
/* 
 * Release Notes: 
 * BSIM3v3.1,   Released by yuhua  96/12/08
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3v1def.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM3v1convTest(GENmodel *inModel, CKTcircuit *ckt)
{
BSIM3v1model *model = (BSIM3v1model*)inModel;
BSIM3v1instance *here;
double delvbd, delvbs, delvds, delvgd, delvgs, vbd, vbs, vds;
double cbd, cbhat, cbs, cd, cdhat, tol, vgd, vgdo, vgs;

    /*  loop through all the BSIM3v1 device models */
    for (; model != NULL; model = BSIM3v1nextModel(model))
    {    /* loop through all the instances of the model */
         for (here = BSIM3v1instances(model); here != NULL ;
              here=BSIM3v1nextInstance(here)) 
	 {
	      vbs = model->BSIM3v1type 
		  * (*(ckt->CKTrhsOld+here->BSIM3v1bNode) 
		  - *(ckt->CKTrhsOld+here->BSIM3v1sNodePrime));
              vgs = model->BSIM3v1type
		  * (*(ckt->CKTrhsOld+here->BSIM3v1gNode) 
		  - *(ckt->CKTrhsOld+here->BSIM3v1sNodePrime));
              vds = model->BSIM3v1type
		  * (*(ckt->CKTrhsOld+here->BSIM3v1dNodePrime) 
		  - *(ckt->CKTrhsOld+here->BSIM3v1sNodePrime));
              vbd = vbs - vds;
              vgd = vgs - vds;
              vgdo = *(ckt->CKTstate0 + here->BSIM3v1vgs) 
		   - *(ckt->CKTstate0 + here->BSIM3v1vds);
              delvbs = vbs - *(ckt->CKTstate0 + here->BSIM3v1vbs);
              delvbd = vbd - *(ckt->CKTstate0 + here->BSIM3v1vbd);
              delvgs = vgs - *(ckt->CKTstate0 + here->BSIM3v1vgs);
              delvds = vds - *(ckt->CKTstate0 + here->BSIM3v1vds);
              delvgd = vgd-vgdo;

              cd = here->BSIM3v1cd;
              if (here->BSIM3v1mode >= 0)
	      {   cdhat = cd - here->BSIM3v1gbd * delvbd 
			+ here->BSIM3v1gmbs * delvbs + here->BSIM3v1gm * delvgs
			+ here->BSIM3v1gds * delvds;
              }
	      else
	      {   cdhat = cd - (here->BSIM3v1gbd - here->BSIM3v1gmbs) * delvbd 
			- here->BSIM3v1gm * delvgd + here->BSIM3v1gds * delvds;
              }

            /*
             *  check convergence
             */
              if ((here->BSIM3v1off == 0)  || (!(ckt->CKTmode & MODEINITFIX)))
	      {   tol = ckt->CKTreltol * MAX(fabs(cdhat), fabs(cd))
		      + ckt->CKTabstol;
                  if (fabs(cdhat - cd) >= tol)
		  {   ckt->CKTnoncon++;
                      return(OK);
                  } 
                  cbs = here->BSIM3v1cbs;
                  cbd = here->BSIM3v1cbd;
                  cbhat = cbs + cbd + here->BSIM3v1gbd * delvbd 
		        + here->BSIM3v1gbs * delvbs;
                  tol = ckt->CKTreltol * MAX(fabs(cbhat), fabs(cbs + cbd))
		      + ckt->CKTabstol;
                  if (fabs(cbhat - (cbs + cbd)) > tol) 
		  {   ckt->CKTnoncon++;
                      return(OK);
                  }
              }
         }
    }
    return(OK);
}


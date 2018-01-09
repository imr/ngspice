/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v0cvtest.c
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3v0def.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM3v0convTest(GENmodel *inModel, CKTcircuit *ckt)
{
BSIM3v0model *model = (BSIM3v0model*)inModel;
BSIM3v0instance *here;
double delvbd, delvbs, delvds, delvgd, delvgs, vbd, vbs, vds;
double cbd, cbhat, cbs, cd, cdhat, tol, vgd, vgdo, vgs;

    /*  loop through all the BSIM3v0 device models */
    for (; model != NULL; model = BSIM3v0nextModel(model))
    {    /* loop through all the instances of the model */
         for (here = BSIM3v0instances(model); here != NULL ;
              here=BSIM3v0nextInstance(here)) 
	 {
	      vbs = model->BSIM3v0type 
		  * (*(ckt->CKTrhsOld+here->BSIM3v0bNode) 
		  - *(ckt->CKTrhsOld+here->BSIM3v0sNodePrime));
              vgs = model->BSIM3v0type
		  * (*(ckt->CKTrhsOld+here->BSIM3v0gNode) 
		  - *(ckt->CKTrhsOld+here->BSIM3v0sNodePrime));
              vds = model->BSIM3v0type
		  * (*(ckt->CKTrhsOld+here->BSIM3v0dNodePrime) 
		  - *(ckt->CKTrhsOld+here->BSIM3v0sNodePrime));
              vbd = vbs - vds;
              vgd = vgs - vds;
              vgdo = *(ckt->CKTstate0 + here->BSIM3v0vgs) 
		   - *(ckt->CKTstate0 + here->BSIM3v0vds);
              delvbs = vbs - *(ckt->CKTstate0 + here->BSIM3v0vbs);
              delvbd = vbd - *(ckt->CKTstate0 + here->BSIM3v0vbd);
              delvgs = vgs - *(ckt->CKTstate0 + here->BSIM3v0vgs);
              delvds = vds - *(ckt->CKTstate0 + here->BSIM3v0vds);
              delvgd = vgd-vgdo;

              cd = here->BSIM3v0cd;
              if (here->BSIM3v0mode >= 0)
	      {   cdhat = cd - here->BSIM3v0gbd * delvbd 
			+ here->BSIM3v0gmbs * delvbs + here->BSIM3v0gm * delvgs
			+ here->BSIM3v0gds * delvds;
              }
	      else
	      {   cdhat = cd - (here->BSIM3v0gbd - here->BSIM3v0gmbs) * delvbd 
			- here->BSIM3v0gm * delvgd + here->BSIM3v0gds * delvds;
              }

            /*
             *  check convergence
             */
              if ((here->BSIM3v0off == 0)  || (!(ckt->CKTmode & MODEINITFIX)))
	      {   tol = ckt->CKTreltol * MAX(fabs(cdhat), fabs(cd))
		      + ckt->CKTabstol;
                  if (fabs(cdhat - cd) >= tol)
		  {   ckt->CKTnoncon++;
                      return(OK);
                  } 
                  cbs = here->BSIM3v0cbs;
                  cbd = here->BSIM3v0cbd;
                  cbhat = cbs + cbd + here->BSIM3v0gbd * delvbd 
		        + here->BSIM3v0gbs * delvbs;
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


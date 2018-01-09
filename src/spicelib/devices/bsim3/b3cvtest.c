/**** BSIM3v3.3.0, Released by Xuemei Xi 07/29/2005 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b3cvtest.c of BSIM3v3.3.0
 * Author: 1995 Min-Chie Jeng and Mansun Chan
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001 Xuemei Xi
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3def.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM3convTest(
GENmodel *inModel,
CKTcircuit *ckt)
{
BSIM3model *model = (BSIM3model*)inModel;
BSIM3instance *here;
double delvbd, delvbs, delvds, delvgd, delvgs, vbd, vbs, vds;
double cbd, cbhat, cbs, cd, cdhat, tol, vgd, vgdo, vgs;

    /*  loop through all the BSIM3 device models */
    for (; model != NULL; model = BSIM3nextModel(model))
    {    /* loop through all the instances of the model */
         for (here = BSIM3instances(model); here != NULL ;
              here=BSIM3nextInstance(here)) 
	 {
              vbs = model->BSIM3type 
		  * (*(ckt->CKTrhsOld+here->BSIM3bNode) 
		  - *(ckt->CKTrhsOld+here->BSIM3sNodePrime));
              vgs = model->BSIM3type
		  * (*(ckt->CKTrhsOld+here->BSIM3gNode) 
		  - *(ckt->CKTrhsOld+here->BSIM3sNodePrime));
              vds = model->BSIM3type
		  * (*(ckt->CKTrhsOld+here->BSIM3dNodePrime) 
		  - *(ckt->CKTrhsOld+here->BSIM3sNodePrime));
              vbd = vbs - vds;
              vgd = vgs - vds;
              vgdo = *(ckt->CKTstate0 + here->BSIM3vgs) 
		   - *(ckt->CKTstate0 + here->BSIM3vds);
              delvbs = vbs - *(ckt->CKTstate0 + here->BSIM3vbs);
              delvbd = vbd - *(ckt->CKTstate0 + here->BSIM3vbd);
              delvgs = vgs - *(ckt->CKTstate0 + here->BSIM3vgs);
              delvds = vds - *(ckt->CKTstate0 + here->BSIM3vds);
              delvgd = vgd-vgdo;

              cd = here->BSIM3cd - here->BSIM3cbd;
              if (here->BSIM3mode >= 0)
	      {   cd += here->BSIM3csub;
		  cdhat = cd - here->BSIM3gbd * delvbd 
			+ (here->BSIM3gmbs + here->BSIM3gbbs) * delvbs
			+ (here->BSIM3gm + here->BSIM3gbgs) * delvgs
			+ (here->BSIM3gds + here->BSIM3gbds) * delvds;
              }
	      else
	      {   cdhat = cd + (here->BSIM3gmbs - here->BSIM3gbd) * delvbd 
			+ here->BSIM3gm * delvgd - here->BSIM3gds * delvds;
              }

            /*
             *  check convergence
             */
              if ((here->BSIM3off == 0)  || (!(ckt->CKTmode & MODEINITFIX)))
	      {   tol = ckt->CKTreltol * MAX(fabs(cdhat), fabs(cd))
		      + ckt->CKTabstol;
                  if (fabs(cdhat - cd) >= tol)
		  {   ckt->CKTnoncon++;
                      return(OK);
                  } 
                  cbs = here->BSIM3cbs;
                  cbd = here->BSIM3cbd;
                  if (here->BSIM3mode >= 0)
		  {   cbhat = cbs + cbd - here->BSIM3csub
			    + here->BSIM3gbd * delvbd 
		            + (here->BSIM3gbs - here->BSIM3gbbs) * delvbs
			    - here->BSIM3gbgs * delvgs
			    - here->BSIM3gbds * delvds;
		  }
		  else
		  {   cbhat = cbs + cbd - here->BSIM3csub 
		            + here->BSIM3gbs * delvbs
			    + (here->BSIM3gbd - here->BSIM3gbbs) * delvbd 
			    - here->BSIM3gbgs * delvgd
			    + here->BSIM3gbds * delvds;
		  }
                  tol = ckt->CKTreltol * MAX(fabs(cbhat), 
			fabs(cbs + cbd - here->BSIM3csub)) + ckt->CKTabstol;
                  if (fabs(cbhat - (cbs + cbd - here->BSIM3csub)) > tol) 
		  {   ckt->CKTnoncon++;
                      return(OK);
                  }
              }
         }
    }
    return(OK);
}



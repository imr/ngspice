/* $Id$  */
/*
 $Log$
 Revision 1.1  2000-04-27 20:03:59  pnenzi
 Initial revision

 Revision 1.1.1.1  1999/11/15 10:35:08  root
 Rework imported sources

 Revision 1.2  1999/08/28 21:00:03  manu
 Big commit - merged ngspice.h, misc.h and util.h - protoized fte

 Revision 1.1.1.1  1999/07/30 09:05:13  root
 NG-Spice starting sources

 * Revision 3.2.2 1999/4/20  18:00:00  Weidong
 * BSIM3v3.2.2 release
 *
*/
static char rcsid[] = "$Id$";

/*************************************/

/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Author: 1997-1999 Weidong Liu.
File: b3cvtest.c
**********/

#include "ngspice.h"
#include <stdio.h>
#include <math.h>
#include "cktdefs.h"
#include "bsim3def.h"
#include "trandefs.h"
#include "const.h"
#include "devdefs.h"
#include "sperror.h"
#include "suffix.h"


int
BSIM3convTest(inModel,ckt)
GENmodel *inModel;
register CKTcircuit *ckt;
{
register BSIM3model *model = (BSIM3model*)inModel;
register BSIM3instance *here;
double delvbd, delvbs, delvds, delvgd, delvgs, vbd, vbs, vds;
double cbd, cbhat, cbs, cd, cdhat, tol, vgd, vgdo, vgs;

    /*  loop through all the BSIM3 device models */
    for (; model != NULL; model = model->BSIM3nextModel)
    {    /* loop through all the instances of the model */
         for (here = model->BSIM3instances; here != NULL ;
              here=here->BSIM3nextInstance) 
	 {    
	      if (here->BSIM3owner != ARCHme) continue;
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



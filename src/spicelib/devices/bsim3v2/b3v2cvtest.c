/* $Id$  */
/*
 $Log$
 Revision 1.1.1.1  2000-04-27 20:03:59  pnenzi
 Imported sources

 * Revision 3.2 1998/6/16  18:00:00  Weidong 
 * BSIM3v3.2 release
 *
*/
static char rcsid[] = "$Id$";

/*************************************/

/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v2cvtest.c
**********/

#include "ngspice.h"
#include <stdio.h>
#include <math.h>
#include "cktdefs.h"
#include "bsim3v2def.h"
#include "trandefs.h"
#include "const.h"
#include "devdefs.h"
#include "sperror.h"
#include "suffix.h"


int
BSIM3V2convTest(inModel,ckt)
GENmodel *inModel;
register CKTcircuit *ckt;
{
register BSIM3V2model *model = (BSIM3V2model*)inModel;
register BSIM3V2instance *here;
double delvbd, delvbs, delvds, delvgd, delvgs, vbd, vbs, vds;
double cbd, cbhat, cbs, cd, cdhat, tol, vgd, vgdo, vgs;

    /*  loop through all the BSIM3V2 device models */
    for (; model != NULL; model = model->BSIM3V2nextModel)
    {    /* loop through all the instances of the model */
         for (here = model->BSIM3V2instances; here != NULL ;
              here=here->BSIM3V2nextInstance) 
	 {   
             if (here->BSIM3V2owner != ARCHme) continue; 
              vbs = model->BSIM3V2type 
		  * (*(ckt->CKTrhsOld+here->BSIM3V2bNode) 
		  - *(ckt->CKTrhsOld+here->BSIM3V2sNodePrime));
              vgs = model->BSIM3V2type
		  * (*(ckt->CKTrhsOld+here->BSIM3V2gNode) 
		  - *(ckt->CKTrhsOld+here->BSIM3V2sNodePrime));
              vds = model->BSIM3V2type
		  * (*(ckt->CKTrhsOld+here->BSIM3V2dNodePrime) 
		  - *(ckt->CKTrhsOld+here->BSIM3V2sNodePrime));
              vbd = vbs - vds;
              vgd = vgs - vds;
              vgdo = *(ckt->CKTstate0 + here->BSIM3V2vgs) 
		   - *(ckt->CKTstate0 + here->BSIM3V2vds);
              delvbs = vbs - *(ckt->CKTstate0 + here->BSIM3V2vbs);
              delvbd = vbd - *(ckt->CKTstate0 + here->BSIM3V2vbd);
              delvgs = vgs - *(ckt->CKTstate0 + here->BSIM3V2vgs);
              delvds = vds - *(ckt->CKTstate0 + here->BSIM3V2vds);
              delvgd = vgd-vgdo;

              cd = here->BSIM3V2cd - here->BSIM3V2cbd;
              if (here->BSIM3V2mode >= 0)
	      {   cd += here->BSIM3V2csub;
		  cdhat = cd - here->BSIM3V2gbd * delvbd 
			+ (here->BSIM3V2gmbs + here->BSIM3V2gbbs) * delvbs
			+ (here->BSIM3V2gm + here->BSIM3V2gbgs) * delvgs
			+ (here->BSIM3V2gds + here->BSIM3V2gbds) * delvds;
              }
	      else
	      {   cdhat = cd + (here->BSIM3V2gmbs - here->BSIM3V2gbd) * delvbd 
			+ here->BSIM3V2gm * delvgd - here->BSIM3V2gds * delvds;
              }

            /*
             *  check convergence
             */
              if ((here->BSIM3V2off == 0)  || (!(ckt->CKTmode & MODEINITFIX)))
	      {   tol = ckt->CKTreltol * MAX(fabs(cdhat), fabs(cd))
		      + ckt->CKTabstol;
                  if (fabs(cdhat - cd) >= tol)
		  {   ckt->CKTnoncon++;
                      return(OK);
                  } 
                  cbs = here->BSIM3V2cbs;
                  cbd = here->BSIM3V2cbd;
                  if (here->BSIM3V2mode >= 0)
		  {   cbhat = cbs + cbd - here->BSIM3V2csub
			    + here->BSIM3V2gbd * delvbd 
		            + (here->BSIM3V2gbs - here->BSIM3V2gbbs) * delvbs
			    - here->BSIM3V2gbgs * delvgs
			    - here->BSIM3V2gbds * delvds;
		  }
		  else
		  {   cbhat = cbs + cbd - here->BSIM3V2csub 
		            + here->BSIM3V2gbs * delvbs
			    + (here->BSIM3V2gbd - here->BSIM3V2gbbs) * delvbd 
			    - here->BSIM3V2gbgs * delvgd
			    + here->BSIM3V2gbds * delvds;
		  }
                  tol = ckt->CKTreltol * MAX(fabs(cbhat), 
			fabs(cbs + cbd - here->BSIM3V2csub)) + ckt->CKTabstol;
                  if (fabs(cbhat - (cbs + cbd - here->BSIM3V2csub)) > tol) 
		  {   ckt->CKTnoncon++;
                      return(OK);
                  }
              }
         }
    }
    return(OK);
}



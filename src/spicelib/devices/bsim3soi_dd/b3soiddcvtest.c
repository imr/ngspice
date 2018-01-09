/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soiddcvtest.c          98/5/01
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.1  99/9/27 Pin Su 
 * BSIMDD2.1 release
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b3soidddef.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
B3SOIDDconvTest(GENmodel *inModel, CKTcircuit *ckt)
{
B3SOIDDmodel *model = (B3SOIDDmodel*)inModel;
B3SOIDDinstance *here;
double delvbd, delvbs, delvds, delvgd, delvgs, vbd, vbs, vds;
double cbd, cbhat, cbs, cd, cdhat, tol, vgd, vgdo, vgs;

    /*  loop through all the B3SOIDD device models */
    for (; model != NULL; model = B3SOIDDnextModel(model))
    {    /* loop through all the instances of the model */
         for (here = B3SOIDDinstances(model); here != NULL ;
              here=B3SOIDDnextInstance(here)) 
	 {	 
	      vbs = model->B3SOIDDtype 
		  * (*(ckt->CKTrhsOld+here->B3SOIDDbNode) 
		  - *(ckt->CKTrhsOld+here->B3SOIDDsNodePrime));
              vgs = model->B3SOIDDtype
		  * (*(ckt->CKTrhsOld+here->B3SOIDDgNode) 
		  - *(ckt->CKTrhsOld+here->B3SOIDDsNodePrime));
              vds = model->B3SOIDDtype
		  * (*(ckt->CKTrhsOld+here->B3SOIDDdNodePrime) 
		  - *(ckt->CKTrhsOld+here->B3SOIDDsNodePrime));
              vbd = vbs - vds;
              vgd = vgs - vds;
              vgdo = *(ckt->CKTstate0 + here->B3SOIDDvgs) 
		   - *(ckt->CKTstate0 + here->B3SOIDDvds);
              delvbs = vbs - *(ckt->CKTstate0 + here->B3SOIDDvbs);
              delvbd = vbd - *(ckt->CKTstate0 + here->B3SOIDDvbd);
              delvgs = vgs - *(ckt->CKTstate0 + here->B3SOIDDvgs);
              delvds = vds - *(ckt->CKTstate0 + here->B3SOIDDvds);
              delvgd = vgd-vgdo;

              cd = here->B3SOIDDcd;
              if (here->B3SOIDDmode >= 0)
	      {   cdhat = cd - here->B3SOIDDgjdb * delvbd 
			+ here->B3SOIDDgmbs * delvbs + here->B3SOIDDgm * delvgs
			+ here->B3SOIDDgds * delvds;
              }
	      else
	      {   cdhat = cd - (here->B3SOIDDgjdb - here->B3SOIDDgmbs) * delvbd 
			- here->B3SOIDDgm * delvgd + here->B3SOIDDgds * delvds;
              }

            /*
             *  check convergence
             */
              if ((here->B3SOIDDoff == 0)  || (!(ckt->CKTmode & MODEINITFIX)))
	      {   tol = ckt->CKTreltol * MAX(fabs(cdhat), fabs(cd))
		      + ckt->CKTabstol;
                  if (fabs(cdhat - cd) >= tol)
		  {   ckt->CKTnoncon++;
                      return(OK);
                  } 
                  cbs = here->B3SOIDDcjs;
                  cbd = here->B3SOIDDcjd;
                  cbhat = cbs + cbd + here->B3SOIDDgjdb * delvbd 
		        + here->B3SOIDDgjsb * delvbs;
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


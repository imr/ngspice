/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
Modified by Paolo Nenzi 2002
File: b3soifdcvtest.c          98/5/01
**********/

/*
 * Revision 2.1  99/9/27 Pin Su 
 * BSIMFD2.1 release
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b3soifddef.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
B3SOIFDconvTest(GENmodel *inModel, CKTcircuit *ckt)
{
B3SOIFDmodel *model = (B3SOIFDmodel*)inModel;
B3SOIFDinstance *here;
double delvbd, delvbs, delvds, delvgd, delvgs, vbd, vbs, vds;
double cbd, cbhat, cbs, cd, cdhat, tol, vgd, vgdo, vgs;

    /*  loop through all the B3SOIFD device models */
    for (; model != NULL; model = B3SOIFDnextModel(model))
    {    /* loop through all the instances of the model */
         for (here = B3SOIFDinstances(model); here != NULL ;
              here=B3SOIFDnextInstance(here)) 
	 {	 
	      vbs = model->B3SOIFDtype 
		  * (*(ckt->CKTrhsOld+here->B3SOIFDbNode) 
		  - *(ckt->CKTrhsOld+here->B3SOIFDsNodePrime));
              vgs = model->B3SOIFDtype
		  * (*(ckt->CKTrhsOld+here->B3SOIFDgNode) 
		  - *(ckt->CKTrhsOld+here->B3SOIFDsNodePrime));
              vds = model->B3SOIFDtype
		  * (*(ckt->CKTrhsOld+here->B3SOIFDdNodePrime) 
		  - *(ckt->CKTrhsOld+here->B3SOIFDsNodePrime));
              vbd = vbs - vds;
              vgd = vgs - vds;
              vgdo = *(ckt->CKTstate0 + here->B3SOIFDvgs) 
		   - *(ckt->CKTstate0 + here->B3SOIFDvds);
              delvbs = vbs - *(ckt->CKTstate0 + here->B3SOIFDvbs);
              delvbd = vbd - *(ckt->CKTstate0 + here->B3SOIFDvbd);
              delvgs = vgs - *(ckt->CKTstate0 + here->B3SOIFDvgs);
              delvds = vds - *(ckt->CKTstate0 + here->B3SOIFDvds);
              delvgd = vgd-vgdo;

              cd = here->B3SOIFDcd;
              if (here->B3SOIFDmode >= 0)
	      {   cdhat = cd - here->B3SOIFDgjdb * delvbd 
			+ here->B3SOIFDgmbs * delvbs + here->B3SOIFDgm * delvgs
			+ here->B3SOIFDgds * delvds;
              }
	      else
	      {   cdhat = cd - (here->B3SOIFDgjdb - here->B3SOIFDgmbs) * delvbd 
			- here->B3SOIFDgm * delvgd + here->B3SOIFDgds * delvds;
              }

            /*
             *  check convergence
             */
              if ((here->B3SOIFDoff == 0)  || (!(ckt->CKTmode & MODEINITFIX)))
	      {   tol = ckt->CKTreltol * MAX(fabs(cdhat), fabs(cd))
		      + ckt->CKTabstol;
                  if (fabs(cdhat - cd) >= tol)
		  {   ckt->CKTnoncon++;
                      return(OK);
                  } 
                  cbs = here->B3SOIFDcjs;
                  cbd = here->B3SOIFDcjd;
                  cbhat = cbs + cbd + here->B3SOIFDgjdb * delvbd 
		        + here->B3SOIFDgjsb * delvbs;
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


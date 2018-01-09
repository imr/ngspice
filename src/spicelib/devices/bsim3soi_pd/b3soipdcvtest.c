/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soipdcvtest.c          98/5/01
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.2.3  02/3/5  Pin Su 
 * BSIMPD2.2.3 release
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b3soipddef.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
B3SOIPDconvTest(GENmodel *inModel, CKTcircuit *ckt)
{
B3SOIPDmodel *model = (B3SOIPDmodel*)inModel;
B3SOIPDinstance *here;
double delvbd, delvbs, delvds, delvgd, delvgs, vbd, vbs, vds;
double cbd, cbhat, cbs, cd, cdhat, tol, vgd, vgdo, vgs;

    /*  loop through all the B3SOIPD device models */
    for (; model != NULL; model = B3SOIPDnextModel(model))
    {    /* loop through all the instances of the model */
         for (here = B3SOIPDinstances(model); here != NULL ;
              here=B3SOIPDnextInstance(here)) 
	 {
	      vbs = model->B3SOIPDtype 
		  * (*(ckt->CKTrhsOld+here->B3SOIPDbNode) 
		  - *(ckt->CKTrhsOld+here->B3SOIPDsNodePrime));
              vgs = model->B3SOIPDtype
		  * (*(ckt->CKTrhsOld+here->B3SOIPDgNode) 
		  - *(ckt->CKTrhsOld+here->B3SOIPDsNodePrime));
              vds = model->B3SOIPDtype
		  * (*(ckt->CKTrhsOld+here->B3SOIPDdNodePrime) 
		  - *(ckt->CKTrhsOld+here->B3SOIPDsNodePrime));
              vbd = vbs - vds;
              vgd = vgs - vds;
              vgdo = *(ckt->CKTstate0 + here->B3SOIPDvgs) 
		   - *(ckt->CKTstate0 + here->B3SOIPDvds);
              delvbs = vbs - *(ckt->CKTstate0 + here->B3SOIPDvbs);
              delvbd = vbd - *(ckt->CKTstate0 + here->B3SOIPDvbd);
              delvgs = vgs - *(ckt->CKTstate0 + here->B3SOIPDvgs);
              delvds = vds - *(ckt->CKTstate0 + here->B3SOIPDvds);
              delvgd = vgd-vgdo;

              cd = here->B3SOIPDcd;
              if (here->B3SOIPDmode >= 0)
	      {   cdhat = cd - here->B3SOIPDgjdb * delvbd 
			+ here->B3SOIPDgmbs * delvbs + here->B3SOIPDgm * delvgs
			+ here->B3SOIPDgds * delvds;
              }
	      else
	      {   cdhat = cd - (here->B3SOIPDgjdb - here->B3SOIPDgmbs) * delvbd 
			- here->B3SOIPDgm * delvgd + here->B3SOIPDgds * delvds;
              }

            /*
             *  check convergence
             */
              if ((here->B3SOIPDoff == 0)  || (!(ckt->CKTmode & MODEINITFIX)))
	      {   tol = ckt->CKTreltol * MAX(fabs(cdhat), fabs(cd))
		      + ckt->CKTabstol;
                  if (fabs(cdhat - cd) >= tol)
		  {   ckt->CKTnoncon++;
                      return(OK);
                  } 
                  cbs = here->B3SOIPDcjs;
                  cbd = here->B3SOIPDcjd;
                  cbhat = cbs + cbd + here->B3SOIPDgjdb * delvbd 
		        + here->B3SOIPDgjsb * delvbs;
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


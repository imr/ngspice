/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soicvtest.c          98/5/01
Modified by Paolo Nenzi 2002
**********/


#include "ngspice.h"
#include "cktdefs.h"
#include "b3soidef.h"
#include "trandefs.h"
#include "const.h"
#include "devdefs.h"
#include "sperror.h"
#include "suffix.h"


int
B3SOIconvTest(GENmodel *inModel, CKTcircuit *ckt)
{
B3SOImodel *model = (B3SOImodel*)inModel;
B3SOIinstance *here;
double delvbd, delvbs, delvds, delvgd, delvgs, vbd, vbs, vds;
double cbd, cbhat, cbs, cd, cdhat, tol, vgd, vgdo, vgs;

    /*  loop through all the B3SOI device models */
    for (; model != NULL; model = model->B3SOInextModel)
    {    /* loop through all the instances of the model */
         for (here = model->B3SOIinstances; here != NULL ;
              here=here->B3SOInextInstance) 
	 {    
	 
	      if (here->B3SOIowner != ARCHme)
                      continue;

	      vbs = model->B3SOItype 
		  * (*(ckt->CKTrhsOld+here->B3SOIbNode) 
		  - *(ckt->CKTrhsOld+here->B3SOIsNodePrime));
              vgs = model->B3SOItype
		  * (*(ckt->CKTrhsOld+here->B3SOIgNode) 
		  - *(ckt->CKTrhsOld+here->B3SOIsNodePrime));
              vds = model->B3SOItype
		  * (*(ckt->CKTrhsOld+here->B3SOIdNodePrime) 
		  - *(ckt->CKTrhsOld+here->B3SOIsNodePrime));
              vbd = vbs - vds;
              vgd = vgs - vds;
              vgdo = *(ckt->CKTstate0 + here->B3SOIvgs) 
		   - *(ckt->CKTstate0 + here->B3SOIvds);
              delvbs = vbs - *(ckt->CKTstate0 + here->B3SOIvbs);
              delvbd = vbd - *(ckt->CKTstate0 + here->B3SOIvbd);
              delvgs = vgs - *(ckt->CKTstate0 + here->B3SOIvgs);
              delvds = vds - *(ckt->CKTstate0 + here->B3SOIvds);
              delvgd = vgd-vgdo;

              cd = here->B3SOIcd;
              if (here->B3SOImode >= 0)
	      {   cdhat = cd - here->B3SOIgjdb * delvbd 
			+ here->B3SOIgmbs * delvbs + here->B3SOIgm * delvgs
			+ here->B3SOIgds * delvds;
              }
	      else
	      {   cdhat = cd - (here->B3SOIgjdb - here->B3SOIgmbs) * delvbd 
			- here->B3SOIgm * delvgd + here->B3SOIgds * delvds;
              }

            /*
             *  check convergence
             */
              if ((here->B3SOIoff == 0)  || (!(ckt->CKTmode & MODEINITFIX)))
	      {   tol = ckt->CKTreltol * MAX(fabs(cdhat), fabs(cd))
		      + ckt->CKTabstol;
                  if (fabs(cdhat - cd) >= tol)
		  {   ckt->CKTnoncon++;
                      return(OK);
                  } 
                  cbs = here->B3SOIcjs;
                  cbd = here->B3SOIcjd;
                  cbhat = cbs + cbd + here->B3SOIgjdb * delvbd 
		        + here->B3SOIgjsb * delvbs;
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


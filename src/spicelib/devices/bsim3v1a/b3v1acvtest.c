/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Paolo Nenzi 2002
File: b3v1acvtest.c
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "bsim3v1adef.h"
#include "trandefs.h"
#include "const.h"
#include "devdefs.h"
#include "sperror.h"
#include "suffix.h"


int
BSIM3v1AconvTest(GENmodel *inModel, CKTcircuit *ckt)
{
BSIM3v1Amodel *model = (BSIM3v1Amodel*)inModel;
BSIM3v1Ainstance *here;
double delvbd, delvbs, delvds, delvgd, delvgs, vbd, vbs, vds;
double cbd, cbhat, cbs, cd, cdhat, tol, vgd, vgdo, vgs;

    /*  loop through all the BSIM3v1A device models */
    for (; model != NULL; model = model->BSIM3v1AnextModel)
    {    /* loop through all the instances of the model */
         for (here = model->BSIM3v1Ainstances; here != NULL ;
              here=here->BSIM3v1AnextInstance) 
	 {    

	       if (here->BSIM3v1Aowner != ARCHme)
                     continue;
	 
	      vbs = model->BSIM3v1Atype 
		  * (*(ckt->CKTrhsOld+here->BSIM3v1AbNode) 
		  - *(ckt->CKTrhsOld+here->BSIM3v1AsNodePrime));
              vgs = model->BSIM3v1Atype
		  * (*(ckt->CKTrhsOld+here->BSIM3v1AgNode) 
		  - *(ckt->CKTrhsOld+here->BSIM3v1AsNodePrime));
              vds = model->BSIM3v1Atype
		  * (*(ckt->CKTrhsOld+here->BSIM3v1AdNodePrime) 
		  - *(ckt->CKTrhsOld+here->BSIM3v1AsNodePrime));
              vbd = vbs - vds;
              vgd = vgs - vds;
              vgdo = *(ckt->CKTstate0 + here->BSIM3v1Avgs) 
		   - *(ckt->CKTstate0 + here->BSIM3v1Avds);
              delvbs = vbs - *(ckt->CKTstate0 + here->BSIM3v1Avbs);
              delvbd = vbd - *(ckt->CKTstate0 + here->BSIM3v1Avbd);
              delvgs = vgs - *(ckt->CKTstate0 + here->BSIM3v1Avgs);
              delvds = vds - *(ckt->CKTstate0 + here->BSIM3v1Avds);
              delvgd = vgd-vgdo;

              cd = here->BSIM3v1Acd;
              if (here->BSIM3v1Amode >= 0)
	      {   cdhat = cd - here->BSIM3v1Agbd * delvbd 
			+ here->BSIM3v1Agmbs * delvbs + here->BSIM3v1Agm * delvgs
			+ here->BSIM3v1Agds * delvds;
              }
	      else
	      {   cdhat = cd - (here->BSIM3v1Agbd - here->BSIM3v1Agmbs) * delvbd 
			- here->BSIM3v1Agm * delvgd + here->BSIM3v1Agds * delvds;
              }

            /*
             *  check convergence
             */
              if ((here->BSIM3v1Aoff == 0)  || (!(ckt->CKTmode & MODEINITFIX)))
	      {   tol = ckt->CKTreltol * MAX(fabs(cdhat), fabs(cd))
		      + ckt->CKTabstol;
                  if (fabs(cdhat - cd) >= tol)
		  {   ckt->CKTnoncon++;
                      return(OK);
                  } 
                  cbs = here->BSIM3v1Acbs;
                  cbd = here->BSIM3v1Acbd;
                  cbhat = cbs + cbd + here->BSIM3v1Agbd * delvbd 
		        + here->BSIM3v1Agbs * delvbs;
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


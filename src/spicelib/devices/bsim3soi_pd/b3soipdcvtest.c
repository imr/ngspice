/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soipdcvtest.c          98/5/01
**********/


#include "ngspice.h"
#include <stdio.h>
#include <math.h>
#include "cktdefs.h"
#include "b3soipddef.h"
#include "trandefs.h"
#include "const.h"
#include "devdefs.h"
#include "sperror.h"
#include "suffix.h"


int
B3SOIPDconvTest (inModel, ckt)
     GENmodel *inModel;
     register CKTcircuit *ckt;
{
  register B3SOIPDmodel *model = (B3SOIPDmodel *) inModel;
  register B3SOIPDinstance *here;
  double delvbd, delvbs, delvds, delvgd, delvgs, vbd, vbs, vds;
  double cbd, cbhat, cbs, cd, cdhat, tol, vgd, vgdo, vgs;

  /*  loop through all the B3SOIPD device models */
  for (; model != NULL; model = model->B3SOIPDnextModel)
    {				/* loop through all the instances of the model */
      for (here = model->B3SOIPDinstances; here != NULL;
	   here = here->B3SOIPDnextInstance)
	{
	  vbs = model->B3SOIPDtype
	    * (*(ckt->CKTrhsOld + here->B3SOIPDbNode)
	       - *(ckt->CKTrhsOld + here->B3SOIPDsNodePrime));
	  vgs = model->B3SOIPDtype
	    * (*(ckt->CKTrhsOld + here->B3SOIPDgNode)
	       - *(ckt->CKTrhsOld + here->B3SOIPDsNodePrime));
	  vds = model->B3SOIPDtype
	    * (*(ckt->CKTrhsOld + here->B3SOIPDdNodePrime)
	       - *(ckt->CKTrhsOld + here->B3SOIPDsNodePrime));
	  vbd = vbs - vds;
	  vgd = vgs - vds;
	  vgdo = *(ckt->CKTstate0 + here->B3SOIPDvgs)
	    - *(ckt->CKTstate0 + here->B3SOIPDvds);
	  delvbs = vbs - *(ckt->CKTstate0 + here->B3SOIPDvbs);
	  delvbd = vbd - *(ckt->CKTstate0 + here->B3SOIPDvbd);
	  delvgs = vgs - *(ckt->CKTstate0 + here->B3SOIPDvgs);
	  delvds = vds - *(ckt->CKTstate0 + here->B3SOIPDvds);
	  delvgd = vgd - vgdo;

	  cd = here->B3SOIPDcd;
	  if (here->B3SOIPDmode >= 0)
	    {
	      cdhat = cd - here->B3SOIPDgjdb * delvbd
		+ here->B3SOIPDgmbs * delvbs + here->B3SOIPDgm * delvgs
		+ here->B3SOIPDgds * delvds;
	    }
	  else
	    {
	      cdhat = cd - (here->B3SOIPDgjdb - here->B3SOIPDgmbs) * delvbd
		- here->B3SOIPDgm * delvgd + here->B3SOIPDgds * delvds;
	    }

	  /*
	   *  check convergence
	   */
	  if ((here->B3SOIPDoff == 0) || (!(ckt->CKTmode & MODEINITFIX)))
	    {
	      tol = ckt->CKTreltol * MAX (fabs (cdhat), fabs (cd))
		+ ckt->CKTabstol;
	      if (fabs (cdhat - cd) >= tol)
		{
		  ckt->CKTnoncon++;
		  return (OK);
		}
	      cbs = here->B3SOIPDcjs;
	      cbd = here->B3SOIPDcjd;
	      cbhat = cbs + cbd + here->B3SOIPDgjdb * delvbd
		+ here->B3SOIPDgjsb * delvbs;
	      tol = ckt->CKTreltol * MAX (fabs (cbhat), fabs (cbs + cbd))
		+ ckt->CKTabstol;
	      if (fabs (cbhat - (cbs + cbd)) > tol)
		{
		  ckt->CKTnoncon++;
		  return (OK);
		}
	    }
	}
    }
  return (OK);
}

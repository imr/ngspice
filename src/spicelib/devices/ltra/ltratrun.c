/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1990 Jaijeet S. Roychowdhury
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ltradefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
LTRAtrunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)

{
  LTRAmodel *model = (LTRAmodel *) inModel;
  LTRAinstance *here;
  double i1, i2, i3, i4;
  double i5, i6, d1, d2, d3, d4;
  double tmp;
  double tolerance;
  double current_lte=0.0;
  int maxiter = 2, iterations = 0;
  double x, y, change, deriv, deriv_delta;

  /* loop through all the transmission line models */
  for (; model != NULL; model = LTRAnextModel(model)) {
    /* loop through all the instances of the model */
    for (here = LTRAinstances(model); here != NULL;
         here = LTRAnextInstance(here)) {

      switch (model->LTRAspecialCase) {

      case LTRA_MOD_LC:
      case LTRA_MOD_RLC:

	if (model->LTRAstepLimit == LTRA_MOD_STEPLIMIT) {
	  tmp = model->LTRAtd;
	  *timeStep = MIN(*timeStep, tmp);
	} else {
	  i1 = ((*(ckt->CKTrhsOld + here->LTRAposNode2)
		  - *(ckt->CKTrhsOld + here->LTRAnegNode2))
	      * model->LTRAadmit + *(ckt->CKTrhsOld + here->LTRAbrEq2))
	      * model->LTRAattenuation;
	  i2 = (*(here->LTRAv2 + ckt->CKTtimeIndex) *
	      model->LTRAadmit + *(here->LTRAi2 + ckt->CKTtimeIndex))
	      * model->LTRAattenuation;
	  i3 = (*(here->LTRAv2 + ckt->CKTtimeIndex - 1) * model->LTRAadmit +
	      *(here->LTRAi2 + ckt->CKTtimeIndex - 1))
	      * model->LTRAattenuation;
	  i4 = ((*(ckt->CKTrhsOld + here->LTRAposNode1) -
		  *(ckt->CKTrhsOld + here->LTRAnegNode1))
	      * model->LTRAadmit + *(ckt->CKTrhsOld + here->LTRAbrEq1))
	      * model->LTRAattenuation;
	  i5 = (*(here->LTRAv1 + ckt->CKTtimeIndex) * model->LTRAadmit +
	      *(here->LTRAi1 + ckt->CKTtimeIndex))
	      * model->LTRAattenuation;
	  i6 = (*(here->LTRAv1 + ckt->CKTtimeIndex - 1) * model->LTRAadmit +
	      *(here->LTRAi1 + ckt->CKTtimeIndex - 1))
	      * model->LTRAattenuation;
	  /*
	   * d1 = (i1-i2)/ckt->CKTdeltaOld[1]; d2 =
	   * (i2-i3)/ckt->CKTdeltaOld[2]; d3 = (i4-i5)/ckt->CKTdeltaOld[1];
	   * d4 = (i5-i6)/ckt->CKTdeltaOld[2];
	   */
	  d1 = (i1 - i2) / (ckt->CKTtime - *(ckt->CKTtimePoints +
		  ckt->CKTtimeIndex));
	  d2 = (i2 - i3) / (*(ckt->CKTtimePoints + ckt->CKTtimeIndex)
	      - *(ckt->CKTtimePoints + ckt->CKTtimeIndex - 1));
	  d3 = (i4 - i5) / (ckt->CKTtime - *(ckt->CKTtimePoints +
		  ckt->CKTtimeIndex));
	  d4 = (i5 - i6) / (*(ckt->CKTtimePoints + ckt->CKTtimeIndex)
	      - *(ckt->CKTtimePoints + ckt->CKTtimeIndex - 1));

	  if ((fabs(d1 - d2) >= model->LTRAreltol * MAX(fabs(d1), fabs(d2)) +
		  model->LTRAabstol) ||
	      (fabs(d3 - d4) >= model->LTRAreltol * MAX(fabs(d3), fabs(d4)) +
		  model->LTRAabstol)) {
	    /* derivitive changing - need to schedule after delay */
	    /* the PREVIOUS point was the breakpoint */
	    /* the previous timepoint plus the delay */


	    /*
	     * tmp = *(ckt->CKTtimePoints + ckt->CKTtimeIndex) +
	     * model->LTRAtd; the work of a confused mind minus current time
	     * tmp -= ckt->CKTtime;
	     */

	    tmp = model->LTRAtd;

	    *timeStep = MIN(*timeStep, tmp);
	  }
	}
	break;

      case LTRA_MOD_RC:
      case LTRA_MOD_RG:
	break;

      default:
	return (E_BADPARM);
      }




      /*
       * the above was for the parts of the equations that resemble the
       * lossless equations. Now we need to estimate the local truncation
       * error in each of the three convolution equations, and if possible
       * adjust the timestep so that all of them remain within some bound.
       * Unfortunately, the expression for the LTE in a convolution operation
       * is complicated and costly to evaluate; in addition, no explicit
       * inverse exists.
       * 
       * So what we do here (for the moment) is check to see the current error
       * is acceptable. If so, the timestep is not changed. If not, then an
       * estimate is made for the new timestep using a few iterations of the
       * newton-raphson method.
       * 
       * modification: we change the timestep to half its previous value
       */

      if ((model->LTRAspecialCase == LTRA_MOD_RLC) &&
	  (!model->LTRAtruncDontCut)) {
	*timeStep = MIN(*timeStep, model->LTRAmaxSafeStep);
      }
      if (model->LTRAlteConType != LTRA_MOD_NOCONTROL) {
	switch (model->LTRAspecialCase) {

	case LTRA_MOD_RLC:
	case LTRA_MOD_RC:
	  tolerance = ckt->CKTtrtol * (ckt->CKTreltol * (
		  fabs(here->LTRAinput1) + fabs(here->LTRAinput2))
	      + ckt->CKTabstol);

	  current_lte = LTRAlteCalculate(ckt, (GENmodel *) model,
	      (GENinstance *) here, ckt->CKTtime);

	  if (current_lte >= tolerance) {
	    if (model->LTRAtruncNR) {

	      x = ckt->CKTtime;
	      y = current_lte;
	      for (;;) {
		deriv_delta = 0.01 * (x - *(ckt->CKTtimePoints +
			ckt->CKTtimeIndex));

#ifdef LTRADEBUG
		if (deriv_delta <= 0.0)
		  fprintf(stdout, "LTRAtrunc: error: timestep is now less than zero\n");
#endif
		deriv = LTRAlteCalculate(ckt, (GENmodel *) model,
		    (GENinstance *) here,
		    x + deriv_delta) - y;
		deriv /= deriv_delta;
		change = (tolerance - y) / deriv;
		x += change;
		if (maxiter == 0) {
		  if (fabs(change) <= fabs(deriv_delta))
		    break;
		} else {
		  iterations++;
		  if (iterations >= maxiter)
		    break;
		}
		y = LTRAlteCalculate(ckt, (GENmodel *) model,
		    (GENinstance *) here, x);
	      }

	      tmp = x - *(ckt->CKTtimePoints + ckt->CKTtimeIndex);
	      *timeStep = MIN(*timeStep, tmp);
	    } else
	      *timeStep *= 0.5;
	  }
	  break;

	case LTRA_MOD_RG:
	case LTRA_MOD_LC:
	  break;

	default:
	  return (E_BADPARM);
	}
      }
    }
#ifdef LTRADEBUG
    if (*timeStep >= model->LTRAtd) {
      fprintf(stdout, "LTRAtrunc: Warning: Timestep bigger than delay of line %s\n", model->LTRAmodName);
      fflush(stdout);
    }
#endif
  }
  return (OK);
}

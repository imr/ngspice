/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1990 Jaijeet S. Roychowdhury
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ltradefs.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
LTRAload(GENmodel *inModel, CKTcircuit *ckt)
/*
 * load the appropriate values for the current timepoint into the sparse
 * matrix and the right-hand-side vector
 */
{
  LTRAmodel *model = (LTRAmodel *) inModel;
  LTRAinstance *here;
  double t1=0.0, t2=0.0, t3=0.0;
  double qf1=0.0, qf2=0.0, qf3=0.0;
  double lf2=0.0, lf3=0.0;
  double v1d = 0.0, v2d = 0.0, i1d = 0.0, i2d = 0.0;
  double dummy1=0.0, dummy2=0.0;
  int isaved = 0;
  unsigned tdover = 0;
  int i;
  double max = 0.0, min = 0.0;

  /* loop through all the transmission line models */
  for (; model != NULL; model = LTRAnextModel(model)) {

    if (ckt->CKTmode & MODEDC) {
      switch (model->LTRAspecialCase) {

      case LTRA_MOD_RG:
	dummy1 = model->LTRAlength * sqrt(model->LTRAresist *
	    model->LTRAconduct);
	dummy2 = exp(-dummy1);
	dummy1 = exp(dummy1);	/* LTRA warning: may overflow! */
	model->LTRAcoshlrootGR = 0.5 * (dummy1 + dummy2);

	if (model->LTRAconduct <= 1.0e-10) {	/* hack! */
	  model->LTRArRsLrGRorG = model->LTRAlength *
	      model->LTRAresist;
	} else {
	  model->LTRArRsLrGRorG = 0.5 * (dummy1 -
	      dummy2) * sqrt(model->LTRAresist /
	      model->LTRAconduct);
	}

	if (model->LTRAresist <= 1.0e-10) {	/* hack! */
	  model->LTRArGsLrGRorR = model->LTRAlength *
	      model->LTRAconduct;
	} else {
	  model->LTRArGsLrGRorR = 0.5 * (dummy1 -
	      dummy2) * sqrt(model->LTRAconduct /
	      model->LTRAresist);
	}
	break;

      case LTRA_MOD_RC:
      case LTRA_MOD_LC:
      case LTRA_MOD_RLC:

	/* simple resistor-like behaviour */
	/* nothing to set up */
	break;

      default:
	return (E_BADPARM);
      }				/* switch */
    } else {

      if ((ckt->CKTmode & MODEINITTRAN) ||
	  (ckt->CKTmode & MODEINITPRED)) {
	switch (model->LTRAspecialCase) {

	case LTRA_MOD_RLC:
	case LTRA_MOD_LC:

	  if (ckt->CKTtime > model->LTRAtd) {
	    tdover = 1;
	  } else {
	    tdover = 0;
	  }
	default:
	  break;
	}

    /* FALLTHROUGH added to suppress GCC warning due to
     * -Wimplicit-fallthrough flag */
	switch (model->LTRAspecialCase) {
	case LTRA_MOD_RLC:
	  /*
	   * set up lists of values of the functions at the necessary
	   * timepoints.
	   */


	  /*
	   * set up coefficient lists LTRAh1dashCoeffs, LTRAh2Coeffs,
	   * LTRAh3dashCoeffs for current timepoint
	   */

	  /*
	   * NOTE: h1, h2 and h3 here actually refer to h1tilde, h2tilde,
	   * h3tilde in the paper
	   */

	  /*
	   * Note: many function evaluations are saved by doing the following
	   * all together in one procedure
	   */

	  (void)
	      LTRArlcCoeffsSetup(&(model->LTRAh1dashFirstCoeff),
	      &(model->LTRAh2FirstCoeff),
	      &(model->LTRAh3dashFirstCoeff),
	      model->LTRAh1dashCoeffs, model->LTRAh2Coeffs,
	      model->LTRAh3dashCoeffs, model->LTRAmodelListSize,
	      model->LTRAtd, model->LTRAalpha, model->LTRAbeta,
	      ckt->CKTtime, ckt->CKTtimePoints, ckt->CKTtimeIndex,
	      model->LTRAchopReltol, &(model->LTRAauxIndex));
          /* FALLTHROUGH */


	case LTRA_MOD_LC:

	  /* setting up the coefficients for interpolation */
	  if (tdover) {		/* serious hack -fix! */
	    for (i = ckt->CKTtimeIndex; i >= 0; i--) {
	      if (*(ckt->CKTtimePoints + i) <
		  ckt->CKTtime - model->LTRAtd) {
		break;
	      }
	    }
#ifdef LTRADEBUG
	    if (i == ckt->CKTtimeIndex) {
	      fprintf(stdout, "LTRAload: Warning: timestep larger than delay of line\n");
	      fprintf(stdout, "	Time now: %g\n\n", ckt->CKTtime);
	    }
#endif

	    if (i == ckt->CKTtimeIndex)
	      i--;

/*#ifdef LTRADEBUG*/
	    if (i == -1) {
#ifdef LTRADEBUG
	      printf("LTRAload: mistake: cannot find delayed timepoint\n");
		  return E_INTERN;
#else
	    return E_INTERN;
#endif
            }

	    isaved = i;

	    t2 = *(ckt->CKTtimePoints + i);
	    t3 = *(ckt->CKTtimePoints + i + 1);

	    if ((i != 0) && ((model->LTRAhowToInterp ==
			LTRA_MOD_QUADINTERP) || (model->LTRAhowToInterp ==
			LTRA_MOD_MIXEDINTERP))) {
	      /* quadratic interpolation */
	      t1 = *(ckt->CKTtimePoints + i - 1);

	      LTRAquadInterp(ckt->CKTtime - model->LTRAtd,
		  t1, t2, t3, &qf1, &qf2, &qf3);

	    }
	    if ((i == 0) || (model->LTRAhowToInterp ==
		    LTRA_MOD_MIXEDINTERP) || (model->LTRAhowToInterp
		    == LTRA_MOD_LININTERP)) {	/* linear interpolation */

	      LTRAlinInterp(ckt->CKTtime - model->LTRAtd,
		  t2, t3, &lf2, &lf3);
	    }
	  }
	  /* interpolation coefficients set-up */

	  break;

	case LTRA_MOD_RC:

	  /*
	   * set up lists of values of the coefficients at the necessary
	   * timepoints.
	   */


	  /*
	   * set up coefficient lists LTRAh1dashCoeffs, LTRAh2Coeffs,
	   * LTRAh3dashCoeffs for current timepoint
	   */

	  /*
	   * Note: many function evaluations are saved by doing the following
	   * all together in one procedure
	   */


	  (void)
	      LTRArcCoeffsSetup(&(model->LTRAh1dashFirstCoeff),
	      &(model->LTRAh2FirstCoeff),
	      &(model->LTRAh3dashFirstCoeff),
	      model->LTRAh1dashCoeffs, model->LTRAh2Coeffs,
	      model->LTRAh3dashCoeffs, model->LTRAmodelListSize,
	      model->LTRAcByR, model->LTRArclsqr, ckt->CKTtime,
	      ckt->CKTtimePoints, ckt->CKTtimeIndex, model->LTRAchopReltol);

	  break;

	case LTRA_MOD_RG:
	  break;
	default:
	  return (E_BADPARM);
	}
      }
    }
    /* loop through all the instances of the model */
    for (here = LTRAinstances(model); here != NULL;
         here = LTRAnextInstance(here)) {

      if ((ckt->CKTmode & MODEDC) ||
	  (model->LTRAspecialCase == LTRA_MOD_RG)) {

	switch (model->LTRAspecialCase) {

	case LTRA_MOD_RG:
	  *(here->LTRAibr1Pos1Ptr) += 1.0;
	  *(here->LTRAibr1Neg1Ptr) -= 1.0;
	  *(here->LTRAibr1Pos2Ptr) -= model->LTRAcoshlrootGR;
	  *(here->LTRAibr1Neg2Ptr) += model->LTRAcoshlrootGR;
	  *(here->LTRAibr1Ibr2Ptr) += (1 + ckt->CKTgmin) *
	      model->LTRArRsLrGRorG;

	  *(here->LTRAibr2Ibr2Ptr) += model->LTRAcoshlrootGR;
	  *(here->LTRAibr2Pos2Ptr) -= (1 + ckt->CKTgmin) *
	      model->LTRArGsLrGRorR;
	  *(here->LTRAibr2Neg2Ptr) += (1 + ckt->CKTgmin) *
	      model->LTRArGsLrGRorR;
	  *(here->LTRAibr2Ibr1Ptr) += 1.0;

	  *(here->LTRApos1Ibr1Ptr) += 1.0;
	  *(here->LTRAneg1Ibr1Ptr) -= 1.0;
	  *(here->LTRApos2Ibr2Ptr) += 1.0;
	  *(here->LTRAneg2Ibr2Ptr) -= 1.0;

	  here->LTRAinput1 = here->LTRAinput2 = 0.0;

	  /*
	   * Somewhere else, we have fixed the matrix with zero entries so
	   * that SMPpreOrder doesn't have fits
	   */

	  break;

	case LTRA_MOD_LC:
	case LTRA_MOD_RLC:
	case LTRA_MOD_RC:	/* load a simple resistor */

	  *(here->LTRApos1Ibr1Ptr) += 1.0;
	  *(here->LTRAneg1Ibr1Ptr) -= 1.0;
	  *(here->LTRApos2Ibr2Ptr) += 1.0;
	  *(here->LTRAneg2Ibr2Ptr) -= 1.0;

	  *(here->LTRAibr1Ibr1Ptr) += 1.0;
	  *(here->LTRAibr1Ibr2Ptr) += 1.0;
	  *(here->LTRAibr2Pos1Ptr) += 1.0;
	  *(here->LTRAibr2Pos2Ptr) -= 1.0;
	  *(here->LTRAibr2Ibr1Ptr) -= model->LTRAresist * model->LTRAlength;

	  here->LTRAinput1 = here->LTRAinput2 = 0.0;
	  break;


	default:
	  return (E_BADPARM);
	}

      } else {
	/* all cases other than DC or the RG case */

	/* first timepoint after zero */
	if (ckt->CKTmode & MODEINITTRAN) {
	  if (!(ckt->CKTmode & MODEUIC)) {

	    here->LTRAinitVolt1 =
		(*(ckt->CKTrhsOld + here->LTRAposNode1)
		- *(ckt->CKTrhsOld + here->LTRAnegNode1));
	    here->LTRAinitVolt2 =
		(*(ckt->CKTrhsOld + here->LTRAposNode2)
		- *(ckt->CKTrhsOld + here->LTRAnegNode2));
	    here->LTRAinitCur1 = *(ckt->CKTrhsOld + here->LTRAbrEq1);
	    here->LTRAinitCur2 = *(ckt->CKTrhsOld + here->LTRAbrEq2);
	  }
	}
	/* matrix loading - done every time LTRAload is called */
	switch (model->LTRAspecialCase) {

	case LTRA_MOD_RLC:
	  /* loading for convolution parts' first terms */

	  dummy1 = model->LTRAadmit * model->LTRAh1dashFirstCoeff;
	  *(here->LTRAibr1Pos1Ptr) += dummy1;
	  *(here->LTRAibr1Neg1Ptr) -= dummy1;
	  *(here->LTRAibr2Pos2Ptr) += dummy1;
	  *(here->LTRAibr2Neg2Ptr) -= dummy1;
	  /* end loading for convolution parts' first terms */
      /* FALLTHROUGH */

	case LTRA_MOD_LC:
	  /*
	   * this section loads for the parts of the equations that resemble
	   * the lossless equations
	   */

	  *(here->LTRAibr1Pos1Ptr) += model->LTRAadmit;
	  *(here->LTRAibr1Neg1Ptr) -= model->LTRAadmit;
	  *(here->LTRAibr1Ibr1Ptr) -= 1.0;
	  *(here->LTRApos1Ibr1Ptr) += 1.0;
	  *(here->LTRAneg1Ibr1Ptr) -= 1.0;

	  *(here->LTRAibr2Pos2Ptr) += model->LTRAadmit;
	  *(here->LTRAibr2Neg2Ptr) -= model->LTRAadmit;
	  *(here->LTRAibr2Ibr2Ptr) -= 1.0;
	  *(here->LTRApos2Ibr2Ptr) += 1.0;
	  *(here->LTRAneg2Ibr2Ptr) -= 1.0;

	  /* loading for lossless-like parts over */
	  break;

	case LTRA_MOD_RC:

	  /*
	   * this section loads for the parts of the equations that have no
	   * convolution
	   */

	  *(here->LTRAibr1Ibr1Ptr) -= 1.0;
	  *(here->LTRApos1Ibr1Ptr) += 1.0;
	  *(here->LTRAneg1Ibr1Ptr) -= 1.0;

	  *(here->LTRAibr2Ibr2Ptr) -= 1.0;
	  *(here->LTRApos2Ibr2Ptr) += 1.0;
	  *(here->LTRAneg2Ibr2Ptr) -= 1.0;

	  /* loading for non-convolution parts over */
	  /* loading for convolution parts' first terms */

	  dummy1 = model->LTRAh1dashFirstCoeff;
	  *(here->LTRAibr1Pos1Ptr) += dummy1;
	  *(here->LTRAibr1Neg1Ptr) -= dummy1;
	  *(here->LTRAibr2Pos2Ptr) += dummy1;
	  *(here->LTRAibr2Neg2Ptr) -= dummy1;

	  dummy1 = model->LTRAh2FirstCoeff;
	  *(here->LTRAibr1Ibr2Ptr) -= dummy1;
	  *(here->LTRAibr2Ibr1Ptr) -= dummy1;

	  dummy1 = model->LTRAh3dashFirstCoeff;
	  *(here->LTRAibr1Pos2Ptr) -= dummy1;
	  *(here->LTRAibr1Neg2Ptr) += dummy1;
	  *(here->LTRAibr2Pos1Ptr) -= dummy1;
	  *(here->LTRAibr2Neg1Ptr) += dummy1;

	  /* end loading for convolution parts' first terms */


	  break;
	default:
	  return (E_BADPARM);
	}


	/* INITPRED - first NR iteration of each timepoint */
	/* set up LTRAinputs - to go into the RHS of the circuit equations */

	if (ckt->CKTmode & (MODEINITPRED | MODEINITTRAN)) {

	  here->LTRAinput1 = here->LTRAinput2 = 0.0;

	  switch (model->LTRAspecialCase) {

	  case LTRA_MOD_LC:
	  case LTRA_MOD_RLC:

	    if (tdover) {
	      /* have to interpolate values */

	      if ((isaved != 0) &&
		  ((model->LTRAhowToInterp ==
			  LTRA_MOD_QUADINTERP) ||
		      (model->LTRAhowToInterp ==
			  LTRA_MOD_MIXEDINTERP))) {

		v1d = *(here->LTRAv1 + isaved - 1) * qf1
		    + *(here->LTRAv1 + isaved) * qf2
		    + *(here->LTRAv1 + isaved + 1) * qf3;

		max = MAX(*(here->LTRAv1 + isaved - 1),
		    *(here->LTRAv1 + isaved));
		max = MAX(max, *(here->LTRAv1 + isaved + 1));
		min = MIN(*(here->LTRAv1 + isaved - 1),
		    *(here->LTRAv1 + isaved));
		min = MIN(min, *(here->LTRAv1 + isaved + 1));

	      }
	      if ((model->LTRAhowToInterp ==
		      LTRA_MOD_LININTERP) || (isaved == 0) ||
		  ((isaved != 0) &&
		      ((model->LTRAhowToInterp ==
			      LTRA_MOD_QUADINTERP) ||
			  (model->LTRAhowToInterp ==
			      LTRA_MOD_MIXEDINTERP)) &&
		      ((v1d > max) || (v1d < min)))) {


		if ((isaved != 0) &&
		    (model->LTRAhowToInterp ==
			LTRA_MOD_QUADINTERP)) {
#ifdef LTRADEBUG
		  fprintf(stdout, "LTRAload: warning: interpolated v1 is out of range after timepoint %d\n", ckt->CKTtimeIndex);
		  fprintf(stdout, "	values: %1.8g %1.8g %1.8g; interpolated: %1.8g\n",
		      *(here->LTRAv1 + isaved - 1),
		      *(here->LTRAv1 + isaved),
		      *(here->LTRAv1 + isaved + 1),
		      v1d);
		  fprintf(stdout, "	timepoints are: %1.8g %1.8g %1.8g %1.8g\n", t1, t2, t3, ckt->CKTtime - model->LTRAtd);
#endif
		} else {

		  v1d = *(here->LTRAv1 + isaved) * lf2
		      + *(here->LTRAv1 + isaved + 1) *
		      lf3;
		}

	      }
	      if ((isaved != 0) &&
		  ((model->LTRAhowToInterp ==
			  LTRA_MOD_QUADINTERP) ||
		      (model->LTRAhowToInterp ==
			  LTRA_MOD_MIXEDINTERP))) {

		i1d = *(here->LTRAi1 + isaved - 1) * qf1
		    + *(here->LTRAi1 + isaved) * qf2
		    + *(here->LTRAi1 + isaved + 1) * qf3;

		max = MAX(*(here->LTRAi1 + isaved - 1),
		    *(here->LTRAi1 + isaved));
		max = MAX(max, *(here->LTRAi1 + isaved + 1));
		min = MIN(*(here->LTRAi1 + isaved - 1),
		    *(here->LTRAi1 + isaved));
		min = MIN(min, *(here->LTRAi1 + isaved + 1));

	      }
	      if ((model->LTRAhowToInterp ==
		      LTRA_MOD_LININTERP) || (isaved == 0) ||
		  ((isaved != 0) &&
		      ((model->LTRAhowToInterp ==
			      LTRA_MOD_QUADINTERP) ||
			  (model->LTRAhowToInterp ==
			      LTRA_MOD_MIXEDINTERP)) &&
		      ((i1d > max) || (i1d < min)))) {


		if ((isaved != 0) &&
		    (model->LTRAhowToInterp ==
			LTRA_MOD_QUADINTERP)) {
#ifdef LTRADEBUG
		  fprintf(stdout, "LTRAload: warning: interpolated i1 is out of range after timepoint %d\n", ckt->CKTtimeIndex);
		  fprintf(stdout, "	values: %1.8g %1.8g %1.8g; interpolated: %1.8g\n",
		      *(here->LTRAi1 + isaved - 1),
		      *(here->LTRAi1 + isaved),
		      *(here->LTRAi1 + isaved + 1),
		      i1d);
		  fprintf(stdout, "	timepoints are: %1.8g %1.8g %1.8g %1.8g\n", t1, t2, t3, ckt->CKTtime - model->LTRAtd);
#endif
		} else {

		  i1d = *(here->LTRAi1 + isaved) * lf2
		      + *(here->LTRAi1 + isaved + 1) *
		      lf3;
		}

	      }
	      if ((isaved != 0) &&
		  ((model->LTRAhowToInterp ==
			  LTRA_MOD_QUADINTERP) ||
		      (model->LTRAhowToInterp ==
			  LTRA_MOD_MIXEDINTERP))) {

		v2d = *(here->LTRAv2 + isaved - 1) * qf1
		    + *(here->LTRAv2 + isaved) * qf2
		    + *(here->LTRAv2 + isaved + 1) * qf3;

		max = MAX(*(here->LTRAv2 + isaved - 1),
		    *(here->LTRAv2 + isaved));
		max = MAX(max, *(here->LTRAv2 + isaved + 1));
		min = MIN(*(here->LTRAv2 + isaved - 1),
		    *(here->LTRAv2 + isaved));
		min = MIN(min, *(here->LTRAv2 + isaved + 1));

	      }
	      if ((model->LTRAhowToInterp ==
		      LTRA_MOD_LININTERP) || (isaved == 0) ||
		  ((isaved != 0) &&
		      ((model->LTRAhowToInterp ==
			      LTRA_MOD_QUADINTERP) ||
			  (model->LTRAhowToInterp ==
			      LTRA_MOD_MIXEDINTERP)) &&
		      ((v2d > max) || (v2d < min)))) {


		if ((isaved != 0) &&
		    (model->LTRAhowToInterp ==
			LTRA_MOD_QUADINTERP)) {
#ifdef LTRADEBUG
		  fprintf(stdout, "LTRAload: warning: interpolated v2 is out of range after timepoint %d\n", ckt->CKTtimeIndex);
		  fprintf(stdout, "	values: %1.8g %1.8g %1.8g; interpolated: %1.8g\n",
		      *(here->LTRAv2 + isaved - 1),
		      *(here->LTRAv2 + isaved),
		      *(here->LTRAv2 + isaved + 1),
		      v2d);
		  fprintf(stdout, "	timepoints are: %1.8g %1.8g %1.8g %1.8g\n", t1, t2, t3, ckt->CKTtime - model->LTRAtd);
#endif
		} else {

		  v2d = *(here->LTRAv2 + isaved) * lf2
		      + *(here->LTRAv2 + isaved + 1) *
		      lf3;
		}

	      }
	      if ((isaved != 0) &&
		  ((model->LTRAhowToInterp ==
			  LTRA_MOD_QUADINTERP) ||
		      (model->LTRAhowToInterp ==
			  LTRA_MOD_MIXEDINTERP))) {

		i2d = *(here->LTRAi2 + isaved - 1) * qf1
		    + *(here->LTRAi2 + isaved) * qf2
		    + *(here->LTRAi2 + isaved + 1) * qf3;

		max = MAX(*(here->LTRAi2 + isaved - 1),
		    *(here->LTRAi2 + isaved));
		max = MAX(max, *(here->LTRAi2 + isaved + 1));
		min = MIN(*(here->LTRAi2 + isaved - 1),
		    *(here->LTRAi2 + isaved));
		min = MIN(min, *(here->LTRAi2 + isaved + 1));

	      }
	      if ((model->LTRAhowToInterp ==
		      LTRA_MOD_LININTERP) || (isaved == 0) ||
		  ((isaved != 0) &&
		      ((model->LTRAhowToInterp ==
			      LTRA_MOD_QUADINTERP) ||
			  (model->LTRAhowToInterp ==
			      LTRA_MOD_MIXEDINTERP)) &&
		      ((i2d > max) || (i2d < min)))) {


		if ((isaved != 0) &&
		    (model->LTRAhowToInterp ==
			LTRA_MOD_QUADINTERP)) {
#ifdef LTRADEBUG
		  fprintf(stdout, "LTRAload: warning: interpolated i2 is out of range after timepoint %d\n", ckt->CKTtimeIndex);
		  fprintf(stdout, "	values: %1.8g %1.8g %1.8g; interpolated: %1.8g\n",
		      *(here->LTRAi2 + isaved - 1),
		      *(here->LTRAi2 + isaved),
		      *(here->LTRAi2 + isaved + 1),
		      i2d);
		  fprintf(stdout, "	timepoints are: %1.8g %1.8g %1.8g %1.8g\n", t1, t2, t3, ckt->CKTtime - model->LTRAtd);
#endif
		} else {

		  i2d = *(here->LTRAi2 + isaved) * lf2
		      + *(here->LTRAi2 + isaved + 1) *
		      lf3;
		}

	      }
	    }
	    /* interpolation done */
	    break;

	  case LTRA_MOD_RC:
	    break;

	  default:
	    return (E_BADPARM);
	  }

    /* FALLTHROUGH added to suppress GCC warning due to
     * -Wimplicit-fallthrough flag */
	  switch (model->LTRAspecialCase) {
	  case LTRA_MOD_RLC:

	    /* begin convolution parts */

	    /* convolution of h1dash with v1 and v2 */
	    /* the matrix has already been loaded above */

	    dummy1 = dummy2 = 0.0;
	    for (i = /* model->LTRAh1dashIndex */ ckt->CKTtimeIndex; i > 0; i--) {
	      if (*(model->LTRAh1dashCoeffs + i) != 0.0) {
		dummy1 += *(model->LTRAh1dashCoeffs
		    + i) * (*(here->LTRAv1 + i) -
		    here->LTRAinitVolt1);
		dummy2 += *(model->LTRAh1dashCoeffs
		    + i) * (*(here->LTRAv2 + i) -
		    here->LTRAinitVolt2);
	      }
	    }

	    dummy1 += here->LTRAinitVolt1 *
		model->LTRAintH1dash;
	    dummy2 += here->LTRAinitVolt2 *
		model->LTRAintH1dash;

	    dummy1 -= here->LTRAinitVolt1 *
		model->LTRAh1dashFirstCoeff;
	    dummy2 -= here->LTRAinitVolt2 *
		model->LTRAh1dashFirstCoeff;

	    here->LTRAinput1 -= dummy1 * model->LTRAadmit;
	    here->LTRAinput2 -= dummy2 * model->LTRAadmit;

	    /* end convolution of h1dash with v1 and v2 */

	    /* convolution of h2 with i2 and i1 */

	    dummy1 = dummy2 = 0.0;
	    if (tdover) {

	      /* the term for ckt->CKTtime - model->LTRAtd */

	      dummy1 = (i2d - here->LTRAinitCur2) *
		  model->LTRAh2FirstCoeff;
	      dummy2 = (i1d - here->LTRAinitCur1) *
		  model->LTRAh2FirstCoeff;

	      /* the rest of the convolution */

	      for (i = /* model->LTRAh2Index */ model->LTRAauxIndex; i > 0; i--) {

		if (*(model->LTRAh2Coeffs + i) != 0.0) {
		  dummy1 += *(model->LTRAh2Coeffs
		      + i) * (*(here->LTRAi2 + i) -
		      here->LTRAinitCur2);
		  dummy2 += *(model->LTRAh2Coeffs
		      + i) * (*(here->LTRAi1 + i) -
		      here->LTRAinitCur1);
		}
	      }
	    }
	    /* the initial-condition terms */

	    dummy1 += here->LTRAinitCur2 *
		model->LTRAintH2;
	    dummy2 += here->LTRAinitCur1 *
		model->LTRAintH2;

	    here->LTRAinput1 += dummy1;
	    here->LTRAinput2 += dummy2;

	    /* end convolution of h2 with i2 and i1 */

	    /* convolution of h3dash with v2 and v1 */

	    /* the term for ckt->CKTtime - model->LTRAtd */

	    dummy1 = dummy2 = 0.0;
	    if (tdover) {

	      dummy1 = (v2d - here->LTRAinitVolt2) *
		  model->LTRAh3dashFirstCoeff;
	      dummy2 = (v1d - here->LTRAinitVolt1) *
		  model->LTRAh3dashFirstCoeff;

	      /* the rest of the convolution */

	      for (i = /* model->LTRAh3dashIndex */ model->LTRAauxIndex; i > 0; i--) {
		if (*(model->LTRAh3dashCoeffs + i) != 0.0) {
		  dummy1 += *(model->LTRAh3dashCoeffs
		      + i) * (*(here->LTRAv2 + i) -
		      here->LTRAinitVolt2);
		  dummy2 += *(model->LTRAh3dashCoeffs
		      + i) * (*(here->LTRAv1 + i) -
		      here->LTRAinitVolt1);
		}
	      }
	    }
	    /* the initial-condition terms */

	    dummy1 += here->LTRAinitVolt2 *
		model->LTRAintH3dash;
	    dummy2 += here->LTRAinitVolt1 *
		model->LTRAintH3dash;

	    here->LTRAinput1 += model->LTRAadmit * dummy1;
	    here->LTRAinput2 += model->LTRAadmit * dummy2;

	    /* end convolution of h3dash with v2 and v1 */

        /* FALLTHROUGH */
	  case LTRA_MOD_LC:
	    /* begin lossless-like parts */

	    if (!tdover) {

	      here->LTRAinput1 += model->LTRAattenuation *
		  (here->LTRAinitVolt2 * model->LTRAadmit +
		  here->LTRAinitCur2);
	      here->LTRAinput2 += model->LTRAattenuation *
		  (here->LTRAinitVolt1 * model->LTRAadmit +
		  here->LTRAinitCur1);

	    } else {

	      here->LTRAinput1 += model->LTRAattenuation *
		  (v2d * model->LTRAadmit + i2d);
	      here->LTRAinput2 += model->LTRAattenuation *
		  (v1d * model->LTRAadmit + i1d);

	    }

	    /* end lossless-like parts */
	    break;

	  case LTRA_MOD_RC:


	    /* begin convolution parts */

	    /* convolution of h1dash with v1 and v2 */
	    /* the matrix has already been loaded above */

	    dummy1 = 0.0;
	    dummy2 = 0.0;
	    for (i = ckt->CKTtimeIndex; i > 0; i--) {
	      if (*(model->LTRAh1dashCoeffs + i) != 0.0) {
		dummy1 += *(model->LTRAh1dashCoeffs
		    + i) * (*(here->LTRAv1 + i) -
		    here->LTRAinitVolt1);
		dummy2 += *(model->LTRAh1dashCoeffs
		    + i) * (*(here->LTRAv2 + i) -
		    here->LTRAinitVolt2);
	      }
	    }

	    /* the initial condition terms */

	    dummy1 += here->LTRAinitVolt1 *
		model->LTRAintH1dash;
	    dummy2 += here->LTRAinitVolt2 *
		model->LTRAintH1dash;

	    /*
	     * the constant contributed by the init condition and the latest
	     * timepoint
	     */

	    dummy1 -= here->LTRAinitVolt1 *
		model->LTRAh1dashFirstCoeff;
	    dummy2 -= here->LTRAinitVolt2 *
		model->LTRAh1dashFirstCoeff;

	    here->LTRAinput1 -= dummy1;
	    here->LTRAinput2 -= dummy2;

	    /* end convolution of h1dash with v1 and v2 */


	    /* convolution of h2 with i2 and i1 */

	    dummy1 = dummy2 = 0.0;

	    for (i = ckt->CKTtimeIndex; i > 0; i--) {
	      if (*(model->LTRAh2Coeffs + i) != 0.0) {
		dummy1 += *(model->LTRAh2Coeffs
		    + i) * (*(here->LTRAi2 + i) -
		    here->LTRAinitCur2);
		dummy2 += *(model->LTRAh2Coeffs
		    + i) * (*(here->LTRAi1 + i) -
		    here->LTRAinitCur1);
	      }
	    }

	    /* the initial-condition terms */

	    dummy1 += here->LTRAinitCur2 *
		model->LTRAintH2;
	    dummy2 += here->LTRAinitCur1 *
		model->LTRAintH2;

	    dummy1 -= here->LTRAinitCur2 *
		model->LTRAh2FirstCoeff;
	    dummy2 -= here->LTRAinitCur1 *
		model->LTRAh2FirstCoeff;

	    here->LTRAinput1 += dummy1;
	    here->LTRAinput2 += dummy2;

	    /* end convolution of h2 with i2 and i1 */

	    /* convolution of h3dash with v2 and v1 */


	    dummy1 = dummy2 = 0.0;


	    for (i = ckt->CKTtimeIndex; i > 0; i--) {
	      if (*(model->LTRAh3dashCoeffs + i) != 0.0) {
		dummy1 += *(model->LTRAh3dashCoeffs
		    + i) * (*(here->LTRAv2 + i) -
		    here->LTRAinitVolt2);
		dummy2 += *(model->LTRAh3dashCoeffs
		    + i) * (*(here->LTRAv1 + i) -
		    here->LTRAinitVolt1);
	      }
	    }

	    /* the initial-condition terms */

	    dummy1 += here->LTRAinitVolt2 *
		model->LTRAintH3dash;
	    dummy2 += here->LTRAinitVolt1 *
		model->LTRAintH3dash;

	    dummy1 -= here->LTRAinitVolt2 *
		model->LTRAh3dashFirstCoeff;
	    dummy2 -= here->LTRAinitVolt1 *
		model->LTRAh3dashFirstCoeff;

	    here->LTRAinput1 += dummy1;
	    here->LTRAinput2 += dummy2;

	    /* end convolution of h3dash with v2 and v1 */

	    break;

	  default:
	    return (E_BADPARM);
	  }
	}
	/* load the RHS - done every time this routine is called */

	*(ckt->CKTrhs + here->LTRAbrEq1) += here->LTRAinput1;
	*(ckt->CKTrhs + here->LTRAbrEq2) += here->LTRAinput2;

      }
    }
  }
  return (OK);
}

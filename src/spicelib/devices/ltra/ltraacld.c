/**********
Copyright 1990 Regents of the University of California.  All rights
reserved.
Author: 1990 Jaijeet S. Roychowdhury
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ltradefs.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
LTRAacLoad(GENmodel *inModel, CKTcircuit *ckt)
/*
 * load the appropriate values for the current timepoint into the sparse
 * matrix and the right-hand-side vector
 */
{
  LTRAmodel *model = (LTRAmodel *) inModel;
  LTRAinstance *here;
  double y0_r, y0_i, lambda_r, lambda_i, mag, theta;
  double exparg_r, exparg_i, explambda_r, explambda_i;
  double y0exp_r, y0exp_i;
  long savemode;
  int error;

  /*
   * LTRAacLoad - loads for LTRA lines for the s.s. ac case the equations are
   * the following:
   * 
   * Y_0(s) * V_1(s) - I_1(s) = exp(-lambda(s)*length) * (Y_0(s) * V_2(s) +
   * I_2(s)) Y_0(s) * V_2(s) - I_2(s) = exp(-lambda(s)*length) * (Y_0(s) *
   * V_1(s) + I_1(s))
   * 
   * where Y_0(s) and lambda(s) are as follows:
   * 
   * Y_0(s) = sqrt( (sC+G)/(sL+R) ) lambda(s) = sqrt( (sC+G)*(sL+R) )
   * 
   * for the RC, RLC, and LC cases, G=0. The RG case is handled exactly as the
   * DC case, (and the above equations require reformulation because they
   * become identical for the DC case.)
   */

  /* loop through all the transmission line models */
  for (; model != NULL; model = LTRAnextModel(model)) {

    switch (model->LTRAspecialCase) {

    case LTRA_MOD_LC:

      y0_r = model->LTRAadmit;
      y0_i = 0.0;
	  /*lambda_i = model->LTRAtd*ckt->CKTomega;*/
	  lambda_i = sqrt(model->LTRAinduct*model->LTRAcapac) * ckt->CKTomega; /*CDHW*/
      lambda_r = 0.0;
      break;

    case LTRA_MOD_RLC:

      theta = 0.5 * atan(model->LTRAresist / (ckt->CKTomega*model->LTRAinduct));
      mag = sqrt(ckt->CKTomega * model->LTRAcapac /
	  sqrt(model->LTRAresist * model->LTRAresist +
	      ckt->CKTomega * ckt->CKTomega * model->LTRAinduct *
	      model->LTRAinduct));
      y0_r = mag * cos(theta);
      y0_i = mag * sin(theta);

      theta = M_PI / 2 - theta;
      mag *= sqrt(model->LTRAresist * model->LTRAresist +
	  ckt->CKTomega * ckt->CKTomega * model->LTRAinduct *
	  model->LTRAinduct);
      lambda_r = mag * cos(theta);
      lambda_i = mag * sin(theta);
      break;

    case LTRA_MOD_RC:

      y0_r = y0_i = sqrt(0.5 * ckt->CKTomega * model->LTRAcByR);
      lambda_r = lambda_i =
	  sqrt(0.5 * ckt->CKTomega * model->LTRAresist * model->LTRAcapac);
      break;

    case LTRA_MOD_RG:

      savemode = ckt->CKTmode;
      ckt->CKTmode |= MODEDC;
      error = LTRAload(inModel, ckt);
      ckt->CKTmode = savemode;
      return (error);
      break;

    default:
      return (E_BADPARM);
    }

    exparg_r = -lambda_r * model->LTRAlength;
    exparg_i = -lambda_i * model->LTRAlength;
    explambda_r = exp(exparg_r) * cos(exparg_i);
    explambda_i = exp(exparg_r) * sin(exparg_i);
    y0exp_r = y0_r * explambda_r - y0_i * explambda_i;
    y0exp_i = y0_r * explambda_i + y0_i * explambda_r;

    /* loop through all the instances of the model */
    for (here = LTRAinstances(model); here != NULL;
         here = LTRAnextInstance(here)) {

      *(here->LTRAibr1Pos1Ptr + 0) += y0_r;
      *(here->LTRAibr1Pos1Ptr + 1) += y0_i;
      *(here->LTRAibr1Neg1Ptr + 0) -= y0_r;
      *(here->LTRAibr1Neg1Ptr + 1) -= y0_i;

      *(here->LTRAibr1Ibr1Ptr + 0) -= 1.0;

      *(here->LTRAibr1Pos2Ptr + 0) -= y0exp_r;
      *(here->LTRAibr1Pos2Ptr + 1) -= y0exp_i;
      *(here->LTRAibr1Neg2Ptr + 0) += y0exp_r;
      *(here->LTRAibr1Neg2Ptr + 1) += y0exp_i;

      *(here->LTRAibr1Ibr2Ptr + 0) -= explambda_r;
      *(here->LTRAibr1Ibr2Ptr + 1) -= explambda_i;

      *(here->LTRAibr2Pos2Ptr + 0) += y0_r;
      *(here->LTRAibr2Pos2Ptr + 1) += y0_i;
      *(here->LTRAibr2Neg2Ptr + 0) -= y0_r;
      *(here->LTRAibr2Neg2Ptr + 1) -= y0_i;

      *(here->LTRAibr2Ibr2Ptr + 0) -= 1.0;

      *(here->LTRAibr2Pos1Ptr + 0) -= y0exp_r;
      *(here->LTRAibr2Pos1Ptr + 1) -= y0exp_i;
      *(here->LTRAibr2Neg1Ptr + 0) += y0exp_r;
      *(here->LTRAibr2Neg1Ptr + 1) += y0exp_i;

      *(here->LTRAibr2Ibr1Ptr + 0) -= explambda_r;
      *(here->LTRAibr2Ibr1Ptr + 1) -= explambda_i;

      *(here->LTRApos1Ibr1Ptr + 0) += 1.0;
      *(here->LTRAneg1Ibr1Ptr + 0) -= 1.0;
      *(here->LTRApos2Ibr2Ptr + 0) += 1.0;
      *(here->LTRAneg2Ibr2Ptr + 0) -= 1.0;
    }
  }
  return (OK);
}

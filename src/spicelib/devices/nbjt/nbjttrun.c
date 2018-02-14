/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine performs truncation error calculations for NBJTs in the
 * circuit.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "nbjtdefs.h"
#include "ngspice/sperror.h"
#include "../../../ciderlib/oned/onedext.h"
#include "ngspice/cidersupt.h"
#include "ngspice/suffix.h"


int
NBJTtrunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
  register NBJTmodel *model = (NBJTmodel *) inModel;
  register NBJTinstance *inst;
  double deltaNew;
  double deltaNorm[7];
  double startTime;
  int i;

  for (i = 0; i <= ckt->CKTmaxOrder; i++) {
    deltaNorm[i] = ckt->CKTdeltaOld[i] / TNorm;
  }
  for (; model != NULL; model = NBJTnextModel(model)) {
    model->NBJTpInfo->order = ckt->CKTorder;
    model->NBJTpInfo->delta = deltaNorm;
    model->NBJTpInfo->lteCoeff = computeLTECoeff(model->NBJTpInfo);
    for (inst = NBJTinstances(model); inst != NULL;
         inst = NBJTnextInstance(inst)) {

      startTime = SPfrontEnd->IFseconds();
      deltaNew = ONEtrunc(inst->NBJTpDevice, model->NBJTpInfo,
	  ckt->CKTdelta);
      *timeStep = MIN(*timeStep, deltaNew);
      inst->NBJTpDevice->pStats->totalTime[STAT_TRAN] +=
	  SPfrontEnd->IFseconds() - startTime;
    }
  }
  return (OK);
}

/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine performs truncation error calculations for NBJTs in the
 * circuit.
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "nbjtdefs.h"
#include "sperror.h"
#include "../../../ciderlib/oned/onedext.h"
#include "cidersupt.h"
#include "suffix.h"


int
NBJTtrunc(inModel, ckt, timeStep)
  GENmodel *inModel;
  register CKTcircuit *ckt;
  double *timeStep;

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
  for (; model != NULL; model = model->NBJTnextModel) {
    model->NBJTpInfo->order = ckt->CKTorder;
    model->NBJTpInfo->delta = deltaNorm;
    model->NBJTpInfo->lteCoeff = computeLTECoeff(model->NBJTpInfo);
    for (inst = model->NBJTinstances; inst != NULL;
	inst = inst->NBJTnextInstance) {
      if (inst->NBJTowner != ARCHme) continue;

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

/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "numddefs.h"
#include "sperror.h"
#include "../../../ciderlib/oned/onedext.h"
#include "suffix.h"
#include "cidersupt.h"



int
NUMDtrunc(inModel, ckt, timeStep)
  GENmodel *inModel;
  register CKTcircuit *ckt;
  double *timeStep;

{
  register NUMDmodel *model = (NUMDmodel *) inModel;
  register NUMDinstance *inst;
  double deltaNew;
  double deltaNorm[7];
  double startTime;
  int i;

  for (i = 0; i <= ckt->CKTmaxOrder; i++) {
    deltaNorm[i] = ckt->CKTdeltaOld[i] / TNorm;
  }
  for (; model != NULL; model = model->NUMDnextModel) {
    model->NUMDpInfo->order = ckt->CKTorder;
    model->NUMDpInfo->delta = deltaNorm;
    model->NUMDpInfo->lteCoeff = computeLTECoeff(model->NUMDpInfo);
    for (inst = model->NUMDinstances; inst != NULL;
	inst = inst->NUMDnextInstance) {
      if (inst->NUMDowner != ARCHme) continue;

      startTime = SPfrontEnd->IFseconds();
      deltaNew = ONEtrunc(inst->NUMDpDevice, model->NUMDpInfo, ckt->CKTdelta);
      *timeStep = MIN(*timeStep, deltaNew);
      inst->NUMDpDevice->pStats->totalTime[STAT_TRAN] +=
	  SPfrontEnd->IFseconds() - startTime;
    }
  }
  return (OK);
}

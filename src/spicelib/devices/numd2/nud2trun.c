/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "numd2def.h"
#include "sperror.h"
#include "../../../ciderlib/twod/twoddefs.h"
#include "../../../ciderlib/twod/twodext.h"
#include "suffix.h"
#include "cidersupt.h"

int
NUMD2trunc(inModel, ckt, timeStep)
  GENmodel *inModel;
  register CKTcircuit *ckt;
  double *timeStep;
{
  register NUMD2model *model = (NUMD2model *) inModel;
  register NUMD2instance *inst;
  double deltaNew;
  double deltaNorm[7];
  double startTime;
  int i;

  for (i = 0; i <= ckt->CKTmaxOrder; i++) {
    deltaNorm[i] = ckt->CKTdeltaOld[i] / TNorm;
  }

  for (; model != NULL; model = model->NUMD2nextModel) {
    OneCarrier = model->NUMD2methods->METHoneCarrier;
    model->NUMD2pInfo->order = ckt->CKTorder;
    model->NUMD2pInfo->delta = deltaNorm;
    model->NUMD2pInfo->lteCoeff = computeLTECoeff(model->NUMD2pInfo);
    for (inst = model->NUMD2instances; inst != NULL;
	inst = inst->NUMD2nextInstance) {
      if (inst->NUMD2owner != ARCHme) continue;

      startTime = SPfrontEnd->IFseconds();
      deltaNew = TWOtrunc(inst->NUMD2pDevice, model->NUMD2pInfo,
	  ckt->CKTdelta);
      *timeStep = MIN(*timeStep, deltaNew);
      inst->NUMD2pDevice->pStats->totalTime[STAT_TRAN] +=
	  SPfrontEnd->IFseconds() - startTime;
    }
  }
  return (OK);
}
